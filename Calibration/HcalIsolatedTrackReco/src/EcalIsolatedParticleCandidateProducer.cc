// -*- C++ -*-
//
// Package:    EcalIsolatedParticleCandidateProducer
// Class:      EcalIsolatedParticleCandidateProducer
//
/**\class EcalIsolatedParticleCandidateProducer EcalIsolatedParticleCandidateProducer.cc Calibration/EcalIsolatedParticleCandidateProducer/src/EcalIsolatedParticleCandidateProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory Safronov
//         Created:  Thu Jun  7 17:21:58 MSD 2007
//
//

// system include files
#include <cmath>
#include <memory>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "Calibration/HcalIsolatedTrackReco/interface/EcalIsolatedParticleCandidateProducer.h"

//#define EDM_ML_DEBUG

EcalIsolatedParticleCandidateProducer::EcalIsolatedParticleCandidateProducer(const edm::ParameterSet& conf) {
  InConeSize_ = conf.getParameter<double>("EcalInnerConeSize");
  OutConeSize_ = conf.getParameter<double>("EcalOuterConeSize");
  hitCountEthr_ = conf.getParameter<double>("ECHitCountEnergyThreshold");
  hitEthr_ = conf.getParameter<double>("ECHitEnergyThreshold");
  tok_l1tau_ = consumes<l1extra::L1JetParticleCollection>(conf.getParameter<edm::InputTag>("L1eTauJetsSource"));
  tok_hlt_ = consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("L1GTSeedLabel"));
  tok_EB_ = consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EBrecHitCollectionLabel"));
  tok_EE_ = consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EErecHitCollectionLabel"));

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  //register your products
  produces<reco::IsolatedPixelTrackCandidateCollection>();
}

EcalIsolatedParticleCandidateProducer::~EcalIsolatedParticleCandidateProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EcalIsolatedParticleCandidateProducer::produce(edm::StreamID,
                                                    edm::Event& iEvent,
                                                    const edm::EventSetup& iSetup) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get tau";
#endif
  edm::Handle<l1extra::L1JetParticleCollection> l1Taus;
  iEvent.getByToken(tok_l1tau_, l1Taus);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get geom";
#endif
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get ec rechit";
#endif
  edm::Handle<EcalRecHitCollection> ecalEB;
  iEvent.getByToken(tok_EB_, ecalEB);

  edm::Handle<EcalRecHitCollection> ecalEE;
  iEvent.getByToken(tok_EE_, ecalEE);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get l1 trig obj";
#endif

  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
  iEvent.getByToken(tok_hlt_, l1trigobj);

  std::vector<edm::Ref<l1t::TauBxCollection> > l1tauobjref;
  std::vector<edm::Ref<l1t::JetBxCollection> > l1jetobjref;

  l1trigobj->getObjects(trigger::TriggerL1Tau, l1tauobjref);
  l1trigobj->getObjects(trigger::TriggerL1Jet, l1jetobjref);

  double ptTriggered = -10;
  double etaTriggered = -100;
  double phiTriggered = -100;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "find highest pT triggered obj";
#endif
  for (unsigned int p = 0; p < l1tauobjref.size(); p++) {
    if (l1tauobjref[p]->pt() > ptTriggered) {
      ptTriggered = l1tauobjref[p]->pt();
      phiTriggered = l1tauobjref[p]->phi();
      etaTriggered = l1tauobjref[p]->eta();
    }
  }
  for (unsigned int p = 0; p < l1jetobjref.size(); p++) {
    if (l1jetobjref[p]->pt() > ptTriggered) {
      ptTriggered = l1jetobjref[p]->pt();
      phiTriggered = l1jetobjref[p]->phi();
      etaTriggered = l1jetobjref[p]->eta();
    }
  }

  auto iptcCollection = std::make_unique<reco::IsolatedPixelTrackCandidateCollection>();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "loop over l1taus";
#endif
  for (l1extra::L1JetParticleCollection::const_iterator tit = l1Taus->begin(); tit != l1Taus->end(); tit++) {
    double dphi = fabs(tit->phi() - phiTriggered);
    if (dphi > M_PI)
      dphi = 2 * M_PI - dphi;
    double Rseed = sqrt(pow(etaTriggered - tit->eta(), 2) + dphi * dphi);
    if (Rseed < 1.2)
      continue;
    int nhitOut = 0;
    int nhitIn = 0;
    double OutEnergy = 0;
    double InEnergy = 0;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "loops over rechits";
#endif
    for (EcalRecHitCollection::const_iterator eItr = ecalEB->begin(); eItr != ecalEB->end(); eItr++) {
      double phiD, R;
      const GlobalPoint& pos = geo->getPosition(eItr->detid());
      double phihit = pos.phi();
      double etahit = pos.eta();
      phiD = fabs(phihit - tit->phi());
      if (phiD > M_PI)
        phiD = 2 * M_PI - phiD;
      R = sqrt(pow(etahit - tit->eta(), 2) + phiD * phiD);

      if (R < OutConeSize_ && R > InConeSize_ && eItr->energy() > hitCountEthr_) {
        nhitOut++;
      }
      if (R < InConeSize_ && eItr->energy() > hitCountEthr_) {
        nhitIn++;
      }

      if (R < OutConeSize_ && R > InConeSize_ && eItr->energy() > hitEthr_) {
        OutEnergy += eItr->energy();
      }
      if (R < InConeSize_ && eItr->energy() > hitEthr_) {
        InEnergy += eItr->energy();
      }
    }

    for (EcalRecHitCollection::const_iterator eItr = ecalEE->begin(); eItr != ecalEE->end(); eItr++) {
      double phiD, R;
      const GlobalPoint& pos = geo->getPosition(eItr->detid());
      double phihit = pos.phi();
      double etahit = pos.eta();
      phiD = fabs(phihit - tit->phi());
      if (phiD > M_PI)
        phiD = 2 * M_PI - phiD;
      R = sqrt(pow(etahit - tit->eta(), 2) + phiD * phiD);
      if (R < OutConeSize_ && R > InConeSize_ && eItr->energy() > hitCountEthr_) {
        nhitOut++;
      }
      if (R < InConeSize_ && eItr->energy() > hitCountEthr_) {
        nhitIn++;
      }
      if (R < OutConeSize_ && R > InConeSize_ && eItr->energy() > hitEthr_) {
        OutEnergy += eItr->energy();
      }
      if (R < InConeSize_ && eItr->energy() > hitEthr_) {
        InEnergy += eItr->energy();
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "create and push_back candidate";
#endif
    reco::IsolatedPixelTrackCandidate newca(
        l1extra::L1JetParticleRef(l1Taus, tit - l1Taus->begin()), InEnergy, OutEnergy, nhitIn, nhitOut);
    iptcCollection->push_back(newca);
  }

  //Use the ExampleData to create an ExampleData2 which
  // is put into the Event

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "put cand into event";
#endif
  iEvent.put(std::move(iptcCollection));
}
// ------------ method called once each job just before starting event loop  ------------
void EcalIsolatedParticleCandidateProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void EcalIsolatedParticleCandidateProducer::endJob() {}

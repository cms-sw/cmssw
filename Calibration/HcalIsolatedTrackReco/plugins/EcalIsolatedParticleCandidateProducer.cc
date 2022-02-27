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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class EcalIsolatedParticleCandidateProducer : public edm::global::EDProducer<> {
public:
  explicit EcalIsolatedParticleCandidateProducer(const edm::ParameterSet&);
  ~EcalIsolatedParticleCandidateProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void beginJob() override {}
  void endJob() override {}

  const double InConeSize_;
  const double OutConeSize_;
  const double hitCountEthr_;
  const double hitEthr_;

  const edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_l1tau_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_hlt_;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;

  // ----------member data ---------------------------
};

EcalIsolatedParticleCandidateProducer::EcalIsolatedParticleCandidateProducer(const edm::ParameterSet& conf)
    : InConeSize_(conf.getParameter<double>("EcalInnerConeSize")),
      OutConeSize_(conf.getParameter<double>("EcalOuterConeSize")),
      hitCountEthr_(conf.getParameter<double>("ECHitCountEnergyThreshold")),
      hitEthr_(conf.getParameter<double>("ECHitEnergyThreshold")),
      tok_l1tau_(consumes<l1extra::L1JetParticleCollection>(conf.getParameter<edm::InputTag>("L1eTauJetsSource"))),
      tok_hlt_(consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("L1GTSeedLabel"))),
      tok_EB_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EBrecHitCollectionLabel"))),
      tok_EE_(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EErecHitCollectionLabel"))),
      tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {
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
  const edm::Handle<l1extra::L1JetParticleCollection>& l1Taus = iEvent.getHandle(tok_l1tau_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get geom";
#endif
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get ec rechit";
#endif
  const edm::Handle<EcalRecHitCollection>& ecalEB = iEvent.getHandle(tok_EB_);
  const edm::Handle<EcalRecHitCollection>& ecalEE = iEvent.getHandle(tok_EE_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "get l1 trig obj";
#endif

  const edm::Handle<trigger::TriggerFilterObjectWithRefs>& l1trigobj = iEvent.getHandle(tok_hlt_);

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

void EcalIsolatedParticleCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("ECHitEnergyThreshold", 0.05);
  desc.add<edm::InputTag>("L1eTauJetsSource", edm::InputTag("l1extraParticles", "Tau"));
  desc.add<edm::InputTag>("L1GTSeedLabel", edm::InputTag("l1sIsolTrack"));
  desc.add<edm::InputTag>("EBrecHitCollectionLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("EErecHitCollectionLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<double>("ECHitCountEnergyThreshold", 0.5);
  desc.add<double>("EcalInnerConeSize", 0.3);
  desc.add<double>("EcalOuterConeSize", 0.7);
  descriptions.add("ecalIsolPartProd", desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EcalIsolatedParticleCandidateProducer);

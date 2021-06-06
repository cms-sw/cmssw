// -*- C++ -*-
//
// Package:    IsolatedEcalPixelTrackCandidateProducer
// Class:      IsolatedEcalPixelTrackCandidateProducer
//
/**\class IsolatedEcalPixelTrackCandidateProducer IsolatedEcalPixelTrackCandidateProducer.cc Calibration/HcalIsolatedTrackReco/src/IsolatedEcalPixelTrackCandidateProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ruchi Gupta
//         Created:  Thu Feb 11 17:21:58 MSD 2014
// $Id: IsolatedEcalPixelTrackCandidateProducer.cc,v 1.0 2014/02/11 22:25:52 wmtan Exp $
//
//

//#define EDM_ML_DEBUG
// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Calibration/HcalIsolatedTrackReco/interface/IsolatedEcalPixelTrackCandidateProducer.h"

//#define EDM_ML_DEBUG

IsolatedEcalPixelTrackCandidateProducer::IsolatedEcalPixelTrackCandidateProducer(const edm::ParameterSet& conf)
    : tok_ee(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EERecHitSource"))),
      tok_eb(consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("EBRecHitSource"))),
      tok_trigcand(consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("filterLabel"))),
      tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      coneSizeEta0_(conf.getParameter<double>("EcalConeSizeEta0")),
      coneSizeEta1_(conf.getParameter<double>("EcalConeSizeEta1")),
      hitCountEthrEB_(conf.getParameter<double>("EBHitCountEnergyThreshold")),
      hitEthrEB_(conf.getParameter<double>("EBHitEnergyThreshold")),
      fachitCountEE_(conf.getParameter<double>("EEFacHitCountEnergyThreshold")),
      hitEthrEE0_(conf.getParameter<double>("EEHitEnergyThreshold0")),
      hitEthrEE1_(conf.getParameter<double>("EEHitEnergyThreshold1")),
      hitEthrEE2_(conf.getParameter<double>("EEHitEnergyThreshold2")),
      hitEthrEE3_(conf.getParameter<double>("EEHitEnergyThreshold3")) {
  // register the products
  produces<reco::IsolatedPixelTrackCandidateCollection>();
}

IsolatedEcalPixelTrackCandidateProducer::~IsolatedEcalPixelTrackCandidateProducer() {}

void IsolatedEcalPixelTrackCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("filterLabel", edm::InputTag("hltIsolPixelTrackL2Filter"));
  desc.add<edm::InputTag>("EBRecHitSource", edm::InputTag("hltEcalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("EERecHitSource", edm::InputTag("hltEcalRecHit", "EcalRecHitsEE"));
  desc.add<double>("EBHitEnergyThreshold", 0.10);
  desc.add<double>("EBHitCountEnergyThreshold", 0.5);
  desc.add<double>("EEHitEnergyThreshold0", -41.0664);
  desc.add<double>("EEHitEnergyThreshold1", 68.7950);
  desc.add<double>("EEHitEnergyThreshold2", -38.1483);
  desc.add<double>("EEHitEnergyThreshold3", 7.04303);
  desc.add<double>("EEFacHitCountEnergyThreshold", 10.0);
  desc.add<double>("EcalConeSizeEta0", 0.09);
  desc.add<double>("EcalConeSizeEta1", 0.14);
  descriptions.add("isolEcalPixelTrackProd", desc);
}

// ------------ method called to produce the data  ------------
void IsolatedEcalPixelTrackCandidateProducer::produce(edm::StreamID,
                                                      edm::Event& iEvent,
                                                      const edm::EventSetup& iSetup) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "==============Inside IsolatedEcalPixelTrackCandidateProducer";
#endif
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);

  edm::Handle<EcalRecHitCollection> ecalEB;
  iEvent.getByToken(tok_eb, ecalEB);

  edm::Handle<EcalRecHitCollection> ecalEE;
  iEvent.getByToken(tok_ee, ecalEE);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "ecal Collections isValid: " << ecalEB.isValid() << "/" << ecalEE.isValid();
#endif

  edm::Handle<trigger::TriggerFilterObjectWithRefs> trigCand;
  iEvent.getByToken(tok_trigcand, trigCand);

  std::vector<edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;
  trigCand->getObjects(trigger::TriggerTrack, isoPixTrackRefs);
  int nCand = isoPixTrackRefs.size();

  auto iptcCollection = std::make_unique<reco::IsolatedPixelTrackCandidateCollection>();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "coneSize_ " << coneSizeEta0_ << "/" << coneSizeEta1_ << " hitCountEthrEB_ "
                                   << hitCountEthrEB_ << " hitEthrEB_ " << hitEthrEB_ << " fachitCountEE_ "
                                   << fachitCountEE_ << " hitEthrEE " << hitEthrEE0_ << ":" << hitEthrEE1_ << ":"
                                   << hitEthrEE2_ << ":" << hitEthrEE3_;
#endif
  for (int p = 0; p < nCand; p++) {
    int nhitIn(0), nhitOut(0);
    double inEnergy(0), outEnergy(0);
    std::pair<double, double> etaPhi(isoPixTrackRefs[p]->track()->eta(), isoPixTrackRefs[p]->track()->phi());
    if (isoPixTrackRefs[p]->etaPhiEcalValid())
      etaPhi = isoPixTrackRefs[p]->etaPhiEcal();
    double etaAbs = std::abs(etaPhi.first);
    double coneSize_ = (etaAbs > 1.5) ? coneSizeEta1_ : (coneSizeEta0_ * (1.5 - etaAbs) + coneSizeEta1_ * etaAbs) / 1.5;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "Track: eta/phi " << etaPhi.first << "/" << etaPhi.second
                                     << " pt:" << isoPixTrackRefs[p]->track()->pt() << " cone " << coneSize_ << "\n"
                                     << "rechit size EB/EE : " << ecalEB->size() << "/" << ecalEE->size()
                                     << " coneSize_: " << coneSize_;
#endif
    if (etaAbs < 1.7) {
      int nin(0), nout(0);
      for (auto eItr : *(ecalEB.product())) {
        const GlobalPoint& pos = geo->getPosition(eItr.detid());
        double R = reco::deltaR(pos.eta(), pos.phi(), etaPhi.first, etaPhi.second);
        if (R < coneSize_) {
          nhitIn++;
          inEnergy += (eItr.energy());
          ++nin;
          if (eItr.energy() > hitCountEthrEB_)
            nhitOut++;
          if (eItr.energy() > hitEthrEB_) {
            outEnergy += (eItr.energy());
            ++nout;
          }
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HcalIsoTrack") << "EBRechit close to the track has E " << eItr.energy()
                                           << " eta/phi: " << pos.eta() << "/" << pos.phi() << " deltaR: " << R;
#endif
        }
      }
    }
    if (etaAbs > 1.25) {
      int nin(0), nout(0);
      for (auto eItr : *(ecalEE.product())) {
        const GlobalPoint& pos = geo->getPosition(eItr.detid());
        double R = reco::deltaR(pos.eta(), pos.phi(), etaPhi.first, etaPhi.second);
        if (R < coneSize_) {
          double eta = std::abs(pos.eta());
          double hitEthr = (((eta * hitEthrEE3_ + hitEthrEE2_) * eta + hitEthrEE1_) * eta + hitEthrEE0_);
          if (hitEthr < hitEthrEB_)
            hitEthr = hitEthrEB_;
          nhitIn++;
          inEnergy += (eItr.energy());
          ++nin;
          if (eItr.energy() > fachitCountEE_ * hitEthr)
            nhitOut++;
          if (eItr.energy() > hitEthr) {
            outEnergy += (eItr.energy());
            ++nout;
          }
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HcalIsoTrack") << "EERechit close to the track has E " << eItr.energy()
                                           << " eta/phi: " << pos.eta() << "/" << pos.phi() << " deltaR: " << R;
#endif
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalIsoTrack") << "nhitIn:" << nhitIn << " inEnergy:" << inEnergy << " nhitOut:" << nhitOut
                                     << " outEnergy:" << outEnergy;
#endif
    reco::IsolatedPixelTrackCandidate newca(*isoPixTrackRefs[p]);
    newca.setEnergyIn(inEnergy);
    newca.setEnergyOut(outEnergy);
    newca.setNHitIn(nhitIn);
    newca.setNHitOut(nhitOut);
    iptcCollection->push_back(newca);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalIsoTrack") << "ncand:" << nCand << " outcollction size:" << iptcCollection->size();
#endif
  iEvent.put(std::move(iptcCollection));
}

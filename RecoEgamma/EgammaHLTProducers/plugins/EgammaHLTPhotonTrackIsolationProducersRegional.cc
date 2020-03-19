// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTPhotonTrackIsolationProducersRegional
//
/**\class EgammaHLTPhotonTrackIsolationProducersRegional EgammaHLTPhotonTrackIsolationProducersRegional.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPhotonTrackIsolationProducersRegional.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTPhotonTrackIsolationProducersRegional.h,v 1.1 2007/03/23 17:22:54 ghezzi Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHLTTrackIsolation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EgammaHLTPhotonTrackIsolationProducersRegional : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTPhotonTrackIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTPhotonTrackIsolationProducersRegional() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<reco::TrackCollection> trackProducer_;

  //edm::ParameterSet conf_;

  const bool countTracks_;

  const double egTrkIsoPtMin_;
  const double egTrkIsoConeSize_;
  const double egTrkIsoZSpan_;
  const double egTrkIsoRSpan_;
  const double egTrkIsoVetoConeSize_;
  const double egTrkIsoStripBarrel_;
  const double egTrkIsoStripEndcap_;

  EgammaHLTTrackIsolation* test_;
};

EgammaHLTPhotonTrackIsolationProducersRegional::EgammaHLTPhotonTrackIsolationProducersRegional(
    const edm::ParameterSet& config)
    : recoEcalCandidateProducer_(
          consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      trackProducer_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("trackProducer"))),
      countTracks_(config.getParameter<bool>("countTracks")),
      egTrkIsoPtMin_(config.getParameter<double>("egTrkIsoPtMin")),
      egTrkIsoConeSize_(config.getParameter<double>("egTrkIsoConeSize")),
      egTrkIsoZSpan_(config.getParameter<double>("egTrkIsoZSpan")),
      egTrkIsoRSpan_(config.getParameter<double>("egTrkIsoRSpan")),
      egTrkIsoVetoConeSize_(config.getParameter<double>("egTrkIsoVetoConeSize")),
      egTrkIsoStripBarrel_(config.getParameter<double>("egTrkIsoStripBarrel")),
      egTrkIsoStripEndcap_(config.getParameter<double>("egTrkIsoStripEndcap")) {
  test_ = new EgammaHLTTrackIsolation(egTrkIsoPtMin_,
                                      egTrkIsoConeSize_,
                                      egTrkIsoZSpan_,
                                      egTrkIsoRSpan_,
                                      egTrkIsoVetoConeSize_,
                                      egTrkIsoStripBarrel_,
                                      egTrkIsoStripEndcap_);

  //register your products
  produces<reco::RecoEcalCandidateIsolationMap>();
}

EgammaHLTPhotonTrackIsolationProducersRegional::~EgammaHLTPhotonTrackIsolationProducersRegional() { delete test_; }

void EgammaHLTPhotonTrackIsolationProducersRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("trackProducer"), edm::InputTag("hltL1SeededEgammaRegionalCTFFinalFitWithMaterial"));
  desc.add<bool>(("countTracks"), false);
  desc.add<double>(("egTrkIsoPtMin"), 1.0);
  desc.add<double>(("egTrkIsoConeSize"), 0.29);
  desc.add<double>(("egTrkIsoZSpan"), 999999.0);
  desc.add<double>(("egTrkIsoRSpan"), 999999.0);
  desc.add<double>(("egTrkIsoVetoConeSize"), 0.06);
  desc.add<double>(("egTrkIsoStripBarrel"), 0.03);
  desc.add<double>(("egTrkIsoStripEndcap"), 0.03);
  descriptions.add(("hltEgammaHLTPhotonTrackIsolationProducersRegional"), desc);
}

// ------------ method called to produce the data  ------------
void EgammaHLTPhotonTrackIsolationProducersRegional::produce(edm::StreamID sid,
                                                             edm::Event& iEvent,
                                                             const edm::EventSetup& iSetup) const {
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_, recoecalcandHandle);

  // Get the barrel hcal hits
  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByToken(trackProducer_, trackHandle);
  const reco::TrackCollection* trackCollection = trackHandle.product();

  reco::RecoEcalCandidateIsolationMap isoMap(recoecalcandHandle);

  for (unsigned int iRecoEcalCand = 0; iRecoEcalCand < recoecalcandHandle->size(); iRecoEcalCand++) {
    reco::RecoEcalCandidateRef recoecalcandref(recoecalcandHandle, iRecoEcalCand);

    bool usePhotonVertex = false;

    float isol;
    if (countTracks_) {
      isol = test_->photonTrackCount(&(*recoecalcandref), trackCollection, usePhotonVertex);
    } else {
      isol = test_->photonPtSum(&(*recoecalcandref), trackCollection, usePhotonVertex);
    }

    isoMap.insert(recoecalcandref, isol);
  }

  iEvent.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(isoMap));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTPhotonTrackIsolationProducersRegional);

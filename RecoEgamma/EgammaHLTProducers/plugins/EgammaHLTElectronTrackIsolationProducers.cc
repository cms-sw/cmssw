// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronTrackIsolationProducers
//
/**\class EgammaHLTElectronTrackIsolationProducers EgammaHLTElectronTrackIsolationProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronTrackIsolationProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//
// $Id: EgammaHLTElectronTrackIsolationProducers.h,v 1.3 2011/12/19 11:16:45 sani Exp $
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

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EgammaHLTElectronTrackIsolationProducers : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet&);
  ~EgammaHLTElectronTrackIsolationProducers() override;
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  const edm::EDGetTokenT<reco::TrackCollection> trackProducer_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotProducer_;

  const bool useGsfTrack_;
  const bool useSCRefs_;

  const double egTrkIsoPtMin_;
  const double egTrkIsoConeSize_;
  const double egTrkIsoZSpan_;
  const double egTrkIsoRSpan_;
  const double egTrkIsoVetoConeSizeBarrel_;
  const double egTrkIsoVetoConeSizeEndcap_;
  const double egTrkIsoStripBarrel_;
  const double egTrkIsoStripEndcap_;
};

EgammaHLTElectronTrackIsolationProducers::EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet& config)
    : electronProducer_(consumes<reco::ElectronCollection>(config.getParameter<edm::InputTag>("electronProducer"))),
      trackProducer_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("trackProducer"))),
      recoEcalCandidateProducer_(
          consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      beamSpotProducer_(consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("beamSpotProducer"))),
      useGsfTrack_(config.getParameter<bool>("useGsfTrack")),
      useSCRefs_(config.getParameter<bool>("useSCRefs")),
      egTrkIsoPtMin_(config.getParameter<double>("egTrkIsoPtMin")),
      egTrkIsoConeSize_(config.getParameter<double>("egTrkIsoConeSize")),
      egTrkIsoZSpan_(config.getParameter<double>("egTrkIsoZSpan")),
      egTrkIsoRSpan_(config.getParameter<double>("egTrkIsoRSpan")),
      egTrkIsoVetoConeSizeBarrel_(config.getParameter<double>("egTrkIsoVetoConeSizeBarrel")),
      egTrkIsoVetoConeSizeEndcap_(config.getParameter<double>("egTrkIsoVetoConeSizeEndcap")),
      egTrkIsoStripBarrel_(config.getParameter<double>("egTrkIsoStripBarrel")),
      egTrkIsoStripEndcap_(config.getParameter<double>("egTrkIsoStripEndcap")) {
  //register your products
  if (useSCRefs_)
    produces<reco::RecoEcalCandidateIsolationMap>();
  else
    produces<reco::ElectronIsolationMap>();
}

EgammaHLTElectronTrackIsolationProducers::~EgammaHLTElectronTrackIsolationProducers() {}

void EgammaHLTElectronTrackIsolationProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electronProducer", edm::InputTag("hltEleAnyWP80PixelMatchElectronsL1Seeded"));
  desc.add<edm::InputTag>("trackProducer", edm::InputTag("hltL1SeededEgammaRegionalCTFFinalFitWithMaterial"));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag());
  desc.add<edm::InputTag>("beamSpotProducer", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<double>("egTrkIsoPtMin", 1.0);
  desc.add<double>("egTrkIsoConeSize", 0.3);
  desc.add<double>("egTrkIsoZSpan", 0.15);
  desc.add<double>("egTrkIsoRSpan", 999999.0);
  desc.add<double>("egTrkIsoVetoConeSizeBarrel", 0.03);
  desc.add<double>("egTrkIsoVetoConeSizeEndcap", 0.03);
  desc.add<double>("egTrkIsoStripBarrel", 0.03);
  desc.add<double>("egTrkIsoStripEndcap", 0.03);
  desc.add<bool>("useGsfTrack", false);
  desc.add<bool>("useSCRefs", false);

  descriptions.add("hltEgammaHLTElectronTrackIsolationProducers", desc);
}
void EgammaHLTElectronTrackIsolationProducers::produce(edm::StreamID sid,
                                                       edm::Event& iEvent,
                                                       const edm::EventSetup&) const {
  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByToken(electronProducer_, electronHandle);

  // Get the general tracks
  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByToken(trackProducer_, trackHandle);
  const reco::TrackCollection* trackCollection = trackHandle.product();

  reco::ElectronIsolationMap eleMap(electronHandle);

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotProducer_, recoBeamSpotHandle);

  const reco::BeamSpot::Point& beamSpotPosition = recoBeamSpotHandle->position();

  ElectronTkIsolation isoAlgo(egTrkIsoConeSize_,
                              egTrkIsoVetoConeSizeBarrel_,
                              egTrkIsoVetoConeSizeEndcap_,
                              egTrkIsoStripBarrel_,
                              egTrkIsoStripEndcap_,
                              egTrkIsoPtMin_,
                              egTrkIsoZSpan_,
                              egTrkIsoRSpan_,
                              trackCollection,
                              beamSpotPosition);

  if (useSCRefs_) {
    edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
    iEvent.getByToken(recoEcalCandidateProducer_, recoEcalCandHandle);
    reco::RecoEcalCandidateIsolationMap recoEcalCandMap(recoEcalCandHandle);

    for (reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand = recoEcalCandHandle->begin();
         iRecoEcalCand != recoEcalCandHandle->end();
         iRecoEcalCand++) {
      reco::RecoEcalCandidateRef recoEcalCandRef(recoEcalCandHandle, iRecoEcalCand - recoEcalCandHandle->begin());

      reco::ElectronRef eleRef;
      for (reco::ElectronCollection::const_iterator eleIt = electronHandle->begin(); eleIt != electronHandle->end();
           eleIt++) {
        if (eleIt->superCluster() == recoEcalCandRef->superCluster()) {
          eleRef = reco::ElectronRef(electronHandle, eleIt - electronHandle->begin());
          break;
        }
      }
      float isol = 999999;
      if (eleRef.isNonnull()) {
        const reco::Track* eleTrk = useGsfTrack_ ? &*eleRef->gsfTrack() : &*eleRef->track();
        isol = isoAlgo.getIso(eleTrk).second;
      }
      recoEcalCandMap.insert(recoEcalCandRef, isol);
    }  //end reco ecal candidate ref

    iEvent.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandMap));

  } else {  //we are going to loop over electron instead
    for (reco::ElectronCollection::const_iterator iElectron = electronHandle->begin();
         iElectron != electronHandle->end();
         iElectron++) {
      reco::ElectronRef eleRef(reco::ElectronRef(electronHandle, iElectron - electronHandle->begin()));
      const reco::Track* eleTrk = useGsfTrack_ ? &*eleRef->gsfTrack() : &*eleRef->track();
      float isol = isoAlgo.getIso(eleTrk).second;
      eleMap.insert(eleRef, isol);
    }

    iEvent.put(std::make_unique<reco::ElectronIsolationMap>(eleMap));
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTElectronTrackIsolationProducers);

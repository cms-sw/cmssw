// Author: Sam Harper (RAL/CERN)
// L1 track isolation producer for the HLT
// A quick primer on how the E/gamma HLT works w.r.t to ID variables
// 1. the supercluster is the primary object
// 2. superclusters get id variables associated to them via association maps keyed
//    to the supercluster
// However here we also need to read in electron objects as we need to solve for which
// GsfTrack associated to the supercluster to use for the isolation
// The electron producer solves for this and assigns the electron the best GsfTrack
// which we will use for the vz of the electron
// One thing which Swagata Mukherjee pointed out is that we have to be careful of
// is getting a bad GsfTrack with a bad vertex which  will give us a fake vz which then
// leads to a perfectly isolated electron as that random vz is not a vertex

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaL1TkIsolation.h"

#include <memory>

class EgammaHLTEleL1TrackIsolProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTEleL1TrackIsolProducer(const edm::ParameterSet&);
  ~EgammaHLTEleL1TrackIsolProducer() override = default;
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> ecalCandsToken_;
  const edm::EDGetTokenT<reco::ElectronCollection> elesToken_;
  const edm::EDGetTokenT<L1TrackCollection> l1TrksToken_;
  EgammaL1TkIsolation isolAlgo_;
};

EgammaHLTEleL1TrackIsolProducer::EgammaHLTEleL1TrackIsolProducer(const edm::ParameterSet& config)
    : ecalCandsToken_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("ecalCands"))),
      elesToken_(consumes<reco::ElectronCollection>(config.getParameter<edm::InputTag>("eles"))),
      l1TrksToken_(consumes<L1TrackCollection>(config.getParameter<edm::InputTag>("l1Tracks"))),
      isolAlgo_(config.getParameter<edm::ParameterSet>("isolCfg")) {
  produces<reco::RecoEcalCandidateIsolationMap>();
}

void EgammaHLTEleL1TrackIsolProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ecalCands", edm::InputTag("hltEgammaCandidates"));
  desc.add<edm::InputTag>("eles", edm::InputTag("hltEgammaGsfElectrons"));
  desc.add<edm::InputTag>("l1Tracks", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add("isolCfg", EgammaL1TkIsolation::makePSetDescription());
  descriptions.add("hltEgammaHLTEleL1TrackIsolProducer", desc);
}
void EgammaHLTEleL1TrackIsolProducer::produce(edm::StreamID sid,
                                              edm::Event& iEvent,
                                              const edm::EventSetup& iSetup) const {
  auto ecalCands = iEvent.getHandle(ecalCandsToken_);
  auto eles = iEvent.getHandle(elesToken_);
  auto l1Trks = iEvent.getHandle(l1TrksToken_);

  auto recoEcalCandMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(ecalCands);

  for (size_t candNr = 0; candNr < ecalCands->size(); candNr++) {
    reco::RecoEcalCandidateRef recoEcalCandRef(ecalCands, candNr);
    reco::ElectronRef eleRef;
    for (size_t eleNr = 0; eleNr < eles->size(); eleNr++) {
      if ((*eles)[eleNr].superCluster() == recoEcalCandRef->superCluster()) {
        eleRef = reco::ElectronRef(eles, eleNr);
        break;
      }
    }

    float isol =
        eleRef.isNonnull() ? isolAlgo_.calIsol(*eleRef->gsfTrack(), *l1Trks).second : std::numeric_limits<float>::max();

    recoEcalCandMap->insert(recoEcalCandRef, isol);
  }
  iEvent.put(std::move(recoEcalCandMap));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTEleL1TrackIsolProducer);

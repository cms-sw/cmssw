#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"

class LowPtGsfElectronCoreProducer : public edm::global::EDProducer<> {
public:
  explicit LowPtGsfElectronCoreProducer(const edm::ParameterSet& conf);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracksToken_;
  const edm::EDGetTokenT<reco::TrackCollection> ctfTracksToken_;
  const edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > superClusterRefs_;
  const edm::EDPutTokenT<reco::GsfElectronCoreCollection> putToken_;
};

using reco::GsfPFRecTrackCollection;
using reco::SuperClusterRef;

LowPtGsfElectronCoreProducer::LowPtGsfElectronCoreProducer(const edm::ParameterSet& config)
    : gsfPfRecTracksToken_(mayConsume<GsfPFRecTrackCollection>(config.getParameter<edm::InputTag>("gsfPfRecTracks"))),
      ctfTracksToken_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("ctfTracks"))),
      superClusterRefs_(consumes<edm::ValueMap<SuperClusterRef> >(config.getParameter<edm::InputTag>("superClusters"))),
      putToken_(produces<reco::GsfElectronCoreCollection>()) {}

void LowPtGsfElectronCoreProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  // Output collection
  reco::GsfElectronCoreCollection electrons;

  // Init
  auto gsfPfRecTracksHandle = event.getHandle(gsfPfRecTracksToken_);
  auto ctfTracksHandle = event.getHandle(ctfTracksToken_);
  auto const& superClusterRefs = event.get(superClusterRefs_);

  // Create ElectronCore objects
  for (size_t ipfgsf = 0; ipfgsf < gsfPfRecTracksHandle->size(); ++ipfgsf) {
    // Refs to GSF(PF) objects and SC
    reco::GsfPFRecTrackRef pfgsf(gsfPfRecTracksHandle, ipfgsf);

    // Use GsfElectronCore(gsf) constructor and store object via emplace
    electrons.emplace_back(pfgsf->gsfTrackRef());

    // Do not consider ECAL-driven objects
    if (electrons.back().ecalDrivenSeed()) {
      electrons.pop_back();
      continue;
    }

    // Add GSF(PF) track information
    auto ctfpair = egamma::getClosestCtfToGsf(electrons.back().gsfTrack(), ctfTracksHandle);
    electrons.back().setCtfTrack(ctfpair.first, ctfpair.second);

    // Add super cluster information
    electrons.back().setSuperCluster(superClusterRefs[pfgsf]);
  }

  event.emplace(putToken_, std::move(electrons));
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronCoreProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfPfRecTracks", {"lowPtGsfElePfGsfTracks"});
  desc.add<edm::InputTag>("ctfTracks", {"generalTracks"});
  desc.add<edm::InputTag>("superClusters", edm::InputTag("lowPtGsfElectronSuperClusters"));
  descriptions.add("lowPtGsfElectronCores", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronCoreProducer);

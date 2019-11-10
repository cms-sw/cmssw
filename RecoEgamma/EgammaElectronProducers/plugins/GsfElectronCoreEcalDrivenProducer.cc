#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"

class GsfElectronCoreEcalDrivenProducer : public edm::global::EDProducer<> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  explicit GsfElectronCoreEcalDrivenProducer(const edm::ParameterSet& conf);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  void produceEcalDrivenCore(const reco::GsfTrackRef& gsfTrackRef,
                             reco::GsfElectronCoreCollection* electrons,
                             edm::Handle<reco::TrackCollection> const& ctfTracksHandle) const;

  const bool useGsfPfRecTracks_;

  const edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracksToken_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracksToken_;
  const edm::EDGetTokenT<reco::TrackCollection> ctfTracksToken_;
};
using namespace reco;

void GsfElectronCoreEcalDrivenProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfPfRecTracks", {"pfTrackElec"});
  desc.add<edm::InputTag>("gsfTracks", {"electronGsfTracks"});
  desc.add<edm::InputTag>("ctfTracks", {"generalTracks"});
  desc.add<bool>("useGsfPfRecTracks", true);
  descriptions.add("ecalDrivenGsfElectronCores", desc);
}

GsfElectronCoreEcalDrivenProducer::GsfElectronCoreEcalDrivenProducer(const edm::ParameterSet& config)
    : useGsfPfRecTracks_(config.getParameter<bool>("useGsfPfRecTracks")),
      gsfPfRecTracksToken_(
          mayConsume<reco::GsfPFRecTrackCollection>(config.getParameter<edm::InputTag>("gsfPfRecTracks"))),
      gsfTracksToken_(consumes<reco::GsfTrackCollection>(config.getParameter<edm::InputTag>("gsfTracks"))),
      ctfTracksToken_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("ctfTracks"))) {
  produces<reco::GsfElectronCoreCollection>();
}

void GsfElectronCoreEcalDrivenProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  auto gsfTracksHandle = event.getHandle(gsfTracksToken_);
  auto ctfTracksHandle = event.getHandle(ctfTracksToken_);

  // output
  auto electrons = std::make_unique<GsfElectronCoreCollection>();

  // loop on ecal driven tracks
  if (useGsfPfRecTracks_) {
    auto const& gsfPfRecTrackCollection = event.get(gsfPfRecTracksToken_);
    GsfPFRecTrackCollection::const_iterator gsfPfRecTrack;
    for (gsfPfRecTrack = gsfPfRecTrackCollection.begin(); gsfPfRecTrack != gsfPfRecTrackCollection.end();
         ++gsfPfRecTrack) {
      const GsfTrackRef gsfTrackRef = gsfPfRecTrack->gsfTrackRef();
      produceEcalDrivenCore(gsfTrackRef, electrons.get(), ctfTracksHandle);
    }
  } else {
    const GsfTrackCollection* gsfTrackCollection = gsfTracksHandle.product();
    for (unsigned int i = 0; i < gsfTrackCollection->size(); ++i) {
      const GsfTrackRef gsfTrackRef = edm::Ref<GsfTrackCollection>(gsfTracksHandle, i);
      produceEcalDrivenCore(gsfTrackRef, electrons.get(), ctfTracksHandle);
    }
  }

  event.put(std::move(electrons));
}

void GsfElectronCoreEcalDrivenProducer::produceEcalDrivenCore(
    const GsfTrackRef& gsfTrackRef,
    GsfElectronCoreCollection* electrons,
    edm::Handle<reco::TrackCollection> const& ctfTracksHandle) const {
  GsfElectronCore* eleCore = new GsfElectronCore(gsfTrackRef);

  if (!eleCore->ecalDrivenSeed()) {
    delete eleCore;
    return;
  }

  auto ctfpair = egamma::getClosestCtfToGsf(eleCore->gsfTrack(), ctfTracksHandle);
  eleCore->setCtfTrack(ctfpair.first, ctfpair.second);

  edm::RefToBase<TrajectorySeed> seed = gsfTrackRef->extra()->seedRef();
  ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();
  edm::RefToBase<CaloCluster> caloCluster = elseed->caloCluster();
  SuperClusterRef scRef = caloCluster.castTo<SuperClusterRef>();
  if (!scRef.isNull()) {
    eleCore->setSuperCluster(scRef);
    electrons->push_back(*eleCore);
  } else {
    edm::LogWarning("GsfElectronCoreEcalDrivenProducer") << "Seed CaloCluster is not a SuperCluster, unexpected...";
  }

  delete eleCore;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfElectronCoreEcalDrivenProducer);

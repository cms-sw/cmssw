#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronTools.h"

class GEDGsfElectronCoreProducer : public edm::global::EDProducer<> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions &);

  explicit GEDGsfElectronCoreProducer(const edm::ParameterSet &conf);
  void produce(edm::StreamID iStream, edm::Event &, const edm::EventSetup &) const override;

private:
  void produceElectronCore(const reco::PFCandidate &pfCandidate,
                           reco::GsfElectronCoreCollection *electrons,
                           edm::Handle<reco::TrackCollection> const &ctfTracksHandle) const;

  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracksToken_;
  const edm::EDGetTokenT<reco::TrackCollection> ctfTracksToken_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> gedEMUnbiasedToken_;
};

using namespace reco;

void GEDGsfElectronCoreProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfTracks", {"electronGsfTracks"});
  desc.add<edm::InputTag>("ctfTracks", {"generalTracks"});
  desc.add<edm::InputTag>("GEDEMUnbiased", {"particleFlowEGamma"});
  descriptions.add("gedGsfElectronCores", desc);
}

GEDGsfElectronCoreProducer::GEDGsfElectronCoreProducer(const edm::ParameterSet &config)
    : gsfTracksToken_(consumes<reco::GsfTrackCollection>(config.getParameter<edm::InputTag>("gsfTracks"))),
      ctfTracksToken_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("ctfTracks"))),
      gedEMUnbiasedToken_(consumes<reco::PFCandidateCollection>(config.getParameter<edm::InputTag>("GEDEMUnbiased"))) {
  produces<reco::GsfElectronCoreCollection>();
}

void GEDGsfElectronCoreProducer::produce(edm::StreamID iStream, edm::Event &event, const edm::EventSetup &setup) const {
  auto ctfTracksHandle = event.getHandle(ctfTracksToken_);

  // output
  auto electrons = std::make_unique<GsfElectronCoreCollection>();

  for (auto const &pfCand : event.get(gedEMUnbiasedToken_)) {
    produceElectronCore(pfCand, electrons.get(), ctfTracksHandle);
  }

  event.put(std::move(electrons));
}

void GEDGsfElectronCoreProducer::produceElectronCore(const reco::PFCandidate &pfCandidate,
                                                     reco::GsfElectronCoreCollection *electrons,
                                                     edm::Handle<reco::TrackCollection> const &ctfTracksHandle) const {
  const GsfTrackRef gsfTrackRef = pfCandidate.gsfTrackRef();
  if (gsfTrackRef.isNull())
    return;

  reco::PFCandidateEGammaExtraRef extraRef = pfCandidate.egammaExtraRef();
  if (extraRef.isNull())
    return;

  GsfElectronCore *eleCore = new GsfElectronCore(gsfTrackRef);

  auto ctfpair = egamma::getClosestCtfToGsf(eleCore->gsfTrack(), ctfTracksHandle);
  eleCore->setCtfTrack(ctfpair.first, ctfpair.second);

  SuperClusterRef scRef = extraRef->superClusterRef();
  SuperClusterRef scBoxRef = extraRef->superClusterPFECALRef();

  for (const auto &convref : extraRef->conversionRef()) {
    eleCore->addConversion(convref);
  }

  for (const auto &convref : extraRef->singleLegConversionRef()) {
    eleCore->addOneLegConversion(convref);
  }

  if (!scRef.isNull() || !scBoxRef.isNull()) {
    eleCore->setSuperCluster(scRef);
    eleCore->setParentSuperCluster(scBoxRef);
    electrons->push_back(*eleCore);
  } else {
    edm::LogWarning("GEDGsfElectronCoreProducer")
        << "Both superClusterRef and superClusterBoxRef of pfCandidate.egammaExtraRef() are Null";
  }

  delete eleCore;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDGsfElectronCoreProducer);

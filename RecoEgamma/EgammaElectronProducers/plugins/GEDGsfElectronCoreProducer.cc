#include "CommonTools/Utils/interface/LazyConstructed.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
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
  void produceElectronCore(reco::GsfTrackRef const &gsfTrackRef,
                           reco::PFCandidateEGammaExtraRef const &extraRef,
                           reco::GsfElectronCoreCollection &electrons,
                           edm::Handle<reco::TrackCollection> const &ctfTracksHandle,
                           edm::soa::EtaPhiTableView ctfTrackVariables) const;

  const edm::EDGetTokenT<reco::TrackCollection> ctfTracksToken_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> gedEMUnbiasedToken_;
  const edm::EDPutTokenT<reco::GsfElectronCoreCollection> putToken_;
};

using namespace reco;

void GEDGsfElectronCoreProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ctfTracks", {"generalTracks"});
  desc.add<edm::InputTag>("GEDEMUnbiased", {"particleFlowEGamma"});
  descriptions.add("gedGsfElectronCores", desc);
}

GEDGsfElectronCoreProducer::GEDGsfElectronCoreProducer(const edm::ParameterSet &config)
    : ctfTracksToken_(consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("ctfTracks"))),
      gedEMUnbiasedToken_(consumes<reco::PFCandidateCollection>(config.getParameter<edm::InputTag>("GEDEMUnbiased"))),
      putToken_(produces<reco::GsfElectronCoreCollection>()) {}

void GEDGsfElectronCoreProducer::produce(edm::StreamID iStream, edm::Event &event, const edm::EventSetup &) const {
  auto ctfTracksHandle = event.getHandle(ctfTracksToken_);
  auto ctfTrackVariables = makeLazy<edm::soa::EtaPhiTable>(*ctfTracksHandle);

  // output
  reco::GsfElectronCoreCollection electrons;

  for (auto const &pfCand : event.get(gedEMUnbiasedToken_)) {
    const GsfTrackRef gsfTrackRef = pfCand.gsfTrackRef();
    if (gsfTrackRef.isNull())
      continue;

    reco::PFCandidateEGammaExtraRef extraRef = pfCand.egammaExtraRef();
    if (extraRef.isNull())
      continue;
    produceElectronCore(gsfTrackRef, extraRef, electrons, ctfTracksHandle, ctfTrackVariables.value());
  }

  event.emplace(putToken_, std::move(electrons));
}

void GEDGsfElectronCoreProducer::produceElectronCore(GsfTrackRef const &gsfTrackRef,
                                                     reco::PFCandidateEGammaExtraRef const &extraRef,
                                                     reco::GsfElectronCoreCollection &electrons,
                                                     edm::Handle<reco::TrackCollection> const &ctfTracksHandle,
                                                     edm::soa::EtaPhiTableView ctfTrackVariables) const {
  electrons.emplace_back(gsfTrackRef);
  auto &eleCore = electrons.back();

  auto ctfpair = egamma::getClosestCtfToGsf(eleCore.gsfTrack(), ctfTracksHandle, ctfTrackVariables);
  eleCore.setCtfTrack(ctfpair.first, ctfpair.second);

  SuperClusterRef scRef = extraRef->superClusterRef();
  SuperClusterRef scBoxRef = extraRef->superClusterPFECALRef();

  for (const auto &convref : extraRef->conversionRef()) {
    eleCore.addConversion(convref);
  }

  for (const auto &convref : extraRef->singleLegConversionRef()) {
    eleCore.addOneLegConversion(convref);
  }

  if (!scRef.isNull() || !scBoxRef.isNull()) {
    eleCore.setSuperCluster(scRef);
    eleCore.setParentSuperCluster(scBoxRef);
  } else {
    electrons.pop_back();
    edm::LogWarning("GEDGsfElectronCoreProducer")
        << "Both superClusterRef and superClusterBoxRef of pfCandidate.egammaExtraRef() are Null";
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDGsfElectronCoreProducer);

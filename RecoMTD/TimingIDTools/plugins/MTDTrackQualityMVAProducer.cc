#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoMTD/TimingIDTools/interface/MTDTrackQualityMVA.h"

using namespace std;
using namespace edm;

class MTDTrackQualityMVAProducer : public edm::stream::EDProducer<> {
public:
  MTDTrackQualityMVAProducer(const ParameterSet& pset);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  template <class H, class T>
  void fillValueMap(edm::Event& iEvent,
                    const edm::Handle<H>& handle,
                    const std::vector<T>& vec,
                    const std::string& name) const;

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

private:
  static constexpr char mvaName[] = "mtdQualMVA";

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<reco::TrackCollection> tracksMTDToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> btlMatchChi2Token_;
  edm::EDGetTokenT<reco::BeamSpot> RecBeamSpotToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> btlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> etlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> etlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> mtdTimeToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmamtdTimeToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> npixBarrelToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> npixEndcapToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> outermostHitPositionToken_;

  MTDTrackQualityMVA mva_;
};

MTDTrackQualityMVAProducer::MTDTrackQualityMVAProducer(const ParameterSet& iConfig)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      btlMatchChi2Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("btlMatchChi2Src"))),
      RecBeamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("offlineBS"))),
      btlMatchTimeChi2Token_(
          consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("btlMatchTimeChi2Src"))),
      etlMatchChi2Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("etlMatchChi2Src"))),
      etlMatchTimeChi2Token_(
          consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("etlMatchTimeChi2Src"))),
      mtdTimeToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("mtdTimeSrc"))),
      sigmamtdTimeToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmamtdTimeSrc"))),
      pathLengthToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"))),
      npixBarrelToken_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("npixBarrelSrc"))),
      npixEndcapToken_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("npixEndcapSrc"))),
      outermostHitPositionToken_(
          consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("outermostHitPositionSrc"))),
      mva_(iConfig.getParameter<edm::FileInPath>("qualityBDT_weights_file").fullPath()) {
  produces<edm::ValueMap<float>>(mvaName);
}

// Configuration descriptions
void MTDTrackQualityMVAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"))->setComment("Input tracks collection");
  desc.add<edm::InputTag>("btlMatchChi2Src", edm::InputTag("trackExtenderWithMTD", "btlMatchChi2"))
      ->setComment("BTL Chi2 Matching value Map");
  desc.add<edm::InputTag>("btlMatchTimeChi2Src", edm::InputTag("trackExtenderWithMTD", "btlMatchTimeChi2"))
      ->setComment("BTL Chi2 Matching value Map");
  desc.add<edm::InputTag>("etlMatchChi2Src", edm::InputTag("trackExtenderWithMTD", "etlMatchChi2"))
      ->setComment("ETL Chi2 Matching value Map");
  desc.add<edm::InputTag>("etlMatchTimeChi2Src", edm::InputTag("trackExtenderWithMTD", "etlMatchTimeChi2"))
      ->setComment("ETL Chi2 Matching value Map");
  desc.add<edm::InputTag>("mtdTimeSrc", edm::InputTag("trackExtenderWithMTD", "generalTracktmtd"))
      ->setComment("MTD Time value Map");
  desc.add<edm::InputTag>("sigmamtdTimeSrc", edm::InputTag("trackExtenderWithMTD", "generalTracksigmatmtd"))
      ->setComment("sigma MTD Time value Map");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD", "generalTrackPathLength"))
      ->setComment("MTD PathLength value Map");
  desc.add<edm::InputTag>("npixBarrelSrc", edm::InputTag("trackExtenderWithMTD", "npixBarrel"))
      ->setComment("# of Barrel pixel associated to refitted tracks");
  desc.add<edm::InputTag>("npixEndcapSrc", edm::InputTag("trackExtenderWithMTD", "npixEndcap"))
      ->setComment("# of Endcap pixel associated to refitted tracks");
  desc.add<edm::InputTag>("outermostHitPositionSrc",
                          edm::InputTag("trackExtenderWithMTD", "generalTrackOutermostHitPosition"));
  desc.add<edm::InputTag>("offlineBS", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::FileInPath>("qualityBDT_weights_file",
                            edm::FileInPath("RecoMTD/TimingIDTools/data/BDT_nvars_17_d7.xml"))
      ->setComment("Track MTD quality BDT weights");
  descriptions.add("mtdTrackQualityMVAProducer", desc);
}

template <class H, class T>
void MTDTrackQualityMVAProducer::fillValueMap(edm::Event& iEvent,
                                              const edm::Handle<H>& handle,
                                              const std::vector<T>& vec,
                                              const std::string& name) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(std::move(out), name);
}

void MTDTrackQualityMVAProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<reco::TrackCollection> tracksH;
  ev.getByToken(tracksToken_, tracksH);
  const auto& tracks = *tracksH;

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> BeamSpotH;
  ev.getByToken(RecBeamSpotToken_, BeamSpotH);
  beamSpot = *BeamSpotH;

  const auto& btlMatchChi2 = ev.get(btlMatchChi2Token_);
  const auto& btlMatchTimeChi2 = ev.get(btlMatchTimeChi2Token_);
  const auto& etlMatchChi2 = ev.get(etlMatchChi2Token_);
  const auto& etlMatchTimeChi2 = ev.get(etlMatchTimeChi2Token_);
  const auto& pathLength = ev.get(pathLengthToken_);
  const auto& npixBarrel = ev.get(npixBarrelToken_);
  const auto& npixEndcap = ev.get(npixEndcapToken_);
  const auto& mtdTime = ev.get(mtdTimeToken_);
  const auto& sigmamtdTime = ev.get(sigmamtdTimeToken_);
  const auto& lHitPos = ev.get(outermostHitPositionToken_);

  std::vector<float> mvaOutRaw;

  //Loop over tracks collection
  for (unsigned int itrack = 0; itrack < tracks.size(); ++itrack) {
    const reco::TrackRef trackref(tracksH, itrack);
    if (pathLength[trackref] == -1.)
      mvaOutRaw.push_back(-1.);
    else {
      mvaOutRaw.push_back(mva_(trackref,
                               beamSpot,
                               npixBarrel,
                               npixEndcap,
                               btlMatchChi2,
                               btlMatchTimeChi2,
                               etlMatchChi2,
                               etlMatchTimeChi2,
                               mtdTime,
                               sigmamtdTime,
                               pathLength,
                               lHitPos));
    }
  }
  fillValueMap(ev, tracksH, mvaOutRaw, mvaName);
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MTDTrackQualityMVAProducer);

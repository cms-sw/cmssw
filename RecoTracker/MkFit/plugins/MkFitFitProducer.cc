#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterfwd.h"

#include "RecoTracker/MkFit/interface/MkFitEventOfHits.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

//CPE
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"

// mkFit includes
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCMS/interface/runFunctions.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/MkBuilderWrapper.h"

// TBB includes
#include "oneapi/tbb/task_arena.h"

// std includes
#include <functional>

class MkFitFitProducer : public edm::global::EDProducer<edm::StreamCache<mkfit::MkBuilderWrapper>> {
public:
  explicit MkFitFitProducer(edm::ParameterSet const& iConfig);
  ~MkFitFitProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<mkfit::MkBuilderWrapper> beginStream(edm::StreamID) const override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<MkFitEventOfHits> eventOfHitsToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<mkfit::IterationConfig, TrackerRecoGeometryRecord> mkFitIterConfigToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCPEToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitOutputWrapper> tracksToken_;
  const bool algoCandCutSelection_;
  const float algoCandMinPtCut_;
  const int algoCandMinNHitsCut_;
  const edm::EDPutTokenT<MkFitOutputWrapper> putToken_;
  const bool mkFitSilent_;
  const bool limitConcurrency_;
};

MkFitFitProducer::MkFitFitProducer(edm::ParameterSet const& iConfig)
    : eventOfHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("eventOfHits"))},
      mkFitGeomToken_{esConsumes()},
      mkFitIterConfigToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("config"))},
      pixelCPEToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")))),
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
      tracksToken_{consumes<MkFitOutputWrapper>(iConfig.getParameter<edm::InputTag>("tracks"))},
      algoCandCutSelection_{bool(iConfig.getParameter<bool>("candCutSel"))},
      algoCandMinPtCut_{float(iConfig.getParameter<double>("candMinPtCut"))},
      algoCandMinNHitsCut_{iConfig.getParameter<int>("candMinNHitsCut")},
      putToken_{produces<MkFitOutputWrapper>()},
      mkFitSilent_{iConfig.getUntrackedParameter<bool>("mkFitSilent")},
      limitConcurrency_{iConfig.getUntrackedParameter<bool>("limitConcurrency")} {
  // TODO: what to do when we have multiple instances of MkFitFitProducer in a job?
  mkfit::MkBuilderWrapper::populate();
}

MkFitFitProducer::~MkFitFitProducer() { mkfit::MkBuilderWrapper::clear(); }

void MkFitFitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("eventOfHits", edm::InputTag("mkFitEventOfHits"));
  desc.add<edm::ESInputTag>("config", edm::ESInputTag(""))
      ->setComment("ESProduct that has the mkFit configuration parameters for this iteration");
  desc.add<std::string>("pixelCPE", "PixelCPETemplateReco");
  desc.add("mkFitPixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("tracks", edm::InputTag{"mkFitProducer"});
  desc.addUntracked("mkFitSilent", true)->setComment("Allows to enables printouts from mkFit with 'False'");
  desc.addUntracked("limitConcurrency", false)
      ->setComment(
          "Use tbb::task_arena to limit the internal concurrency to 1; useful only for timing studies when measuring "
          "the module time");

  //emulate MkFitOutputConverter
  desc.add<bool>("candCutSel", false)->setComment("flag used to trigger cut-based selection at cand level");
  desc.add<double>("candMinPtCut", 0)->setComment("min pt cut at cand level");
  desc.add<int>("candMinNHitsCut", 0)->setComment("min cut on number of hits at cand level");

  descriptions.add("MkFitFitProducerDefault", desc);
}

std::unique_ptr<mkfit::MkBuilderWrapper> MkFitFitProducer::beginStream(edm::StreamID iID) const {
  return std::make_unique<mkfit::MkBuilderWrapper>(mkFitSilent_);
}

void MkFitFitProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const PixelClusterParameterEstimator* pixelCPE = &iSetup.getData(pixelCPEToken_);
  const auto& hits = iEvent.get(pixelClusterIndexToHitToken_).hits();  //const MkFitClusterIndexToHit&

  const auto& eventOfHits = iEvent.get(eventOfHitsToken_);

  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);
  const auto& mkFitIterConfig = iSetup.getData(mkFitIterConfigToken_);

  mkfit::TrackVec tracks;
  auto intracks = iEvent.get(tracksToken_).tracks();

  //emulate MkFitOutputConverter
  if (algoCandCutSelection_) {
    mkfit::TrackVec reducedInput;
    for (auto const& t : intracks) {
      if (!(t.pT() < algoCandMinPtCut_ || t.nTotalHits() < algoCandMinPtCut_))
        reducedInput.push_back(t);
    }
    intracks.swap(reducedInput);
  }

  auto cpe = [&](int orig_hit_idx, float ltp_arr[6], float (&hit_arr)[5]) -> bool {
    auto const& hit = dynamic_cast<SiPixelRecHit const&>(*hits[orig_hit_idx]);
    LocalTrajectoryParameters ltp =
        LocalTrajectoryParameters(ltp_arr[0], ltp_arr[1], ltp_arr[2], ltp_arr[3], ltp_arr[4], ltp_arr[5]);
    const SiPixelCluster& clust = *hit.cluster();
    auto&& params = pixelCPE->getParameters(clust, *hit.detUnit(), ltp);
    //need to check validity and in case return false
    //fill output corr;
    hit_arr[0] = std::get<0>(params).x();
    hit_arr[1] = std::get<0>(params).y();
    hit_arr[2] = std::get<1>(params).xx();
    hit_arr[3] = std::get<1>(params).xy();
    hit_arr[4] = std::get<1>(params).yy();
    return true;
  };

  auto lambda = [&]() {
    mkfit::run_MkFitFit(
        mkFitGeom.trackerInfo(), mkFitIterConfig, eventOfHits.get(), streamCache(iID)->get(), intracks, tracks, cpe);
  };

  if (limitConcurrency_) {
    tbb::task_arena arena(1);
    arena.execute(std::move(lambda));
  } else {
    tbb::this_task_arena::isolate(std::move(lambda));
  }

  iEvent.emplace(putToken_, std::move(tracks), true);
}

DEFINE_FWK_MODULE(MkFitFitProducer);

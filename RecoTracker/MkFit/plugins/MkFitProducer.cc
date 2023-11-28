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

// mkFit includes
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCMS/interface/runFunctions.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/MkBuilderWrapper.h"

// TBB includes
#include "oneapi/tbb/task_arena.h"

// std includes
#include <functional>

class MkFitProducer : public edm::global::EDProducer<edm::StreamCache<mkfit::MkBuilderWrapper>> {
public:
  explicit MkFitProducer(edm::ParameterSet const& iConfig);
  ~MkFitProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<mkfit::MkBuilderWrapper> beginStream(edm::StreamID) const override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  void stripClusterChargeCut(const std::vector<float>& stripClusterCharge, std::vector<bool>& mask) const;

  const edm::EDGetTokenT<MkFitHitWrapper> pixelHitsToken_;
  const edm::EDGetTokenT<MkFitHitWrapper> stripHitsToken_;
  const edm::EDGetTokenT<std::vector<float>> stripClusterChargeToken_;
  const edm::EDGetTokenT<MkFitEventOfHits> eventOfHitsToken_;
  const edm::EDGetTokenT<MkFitSeedWrapper> seedToken_;
  edm::EDGetTokenT<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>> pixelMaskToken_;
  edm::EDGetTokenT<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>> stripMaskToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<mkfit::IterationConfig, TrackerRecoGeometryRecord> mkFitIterConfigToken_;
  const edm::EDPutTokenT<MkFitOutputWrapper> putToken_;
  const float minGoodStripCharge_;
  const bool seedCleaning_;
  const bool backwardFitInCMSSW_;
  const bool removeDuplicates_;
  const bool mkFitSilent_;
  const bool limitConcurrency_;
};

MkFitProducer::MkFitProducer(edm::ParameterSet const& iConfig)
    : pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      stripClusterChargeToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      eventOfHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("eventOfHits"))},
      seedToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
      mkFitGeomToken_{esConsumes()},
      mkFitIterConfigToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("config"))},
      putToken_{produces<MkFitOutputWrapper>()},
      minGoodStripCharge_{static_cast<float>(
          iConfig.getParameter<edm::ParameterSet>("minGoodStripCharge").getParameter<double>("value"))},
      seedCleaning_{iConfig.getParameter<bool>("seedCleaning")},
      backwardFitInCMSSW_{iConfig.getParameter<bool>("backwardFitInCMSSW")},
      removeDuplicates_{iConfig.getParameter<bool>("removeDuplicates")},
      mkFitSilent_{iConfig.getUntrackedParameter<bool>("mkFitSilent")},
      limitConcurrency_{iConfig.getUntrackedParameter<bool>("limitConcurrency")} {
  const auto clustersToSkip = iConfig.getParameter<edm::InputTag>("clustersToSkip");
  if (not clustersToSkip.label().empty()) {
    pixelMaskToken_ = consumes(clustersToSkip);
    stripMaskToken_ = consumes(clustersToSkip);
  }

  const auto build = iConfig.getParameter<std::string>("buildingRoutine");
  if (build == "bestHit") {
    //buildFunction_ = mkfit::runBuildingTestPlexBestHit;
    throw cms::Exception("Configuration") << "bestHit is temporarily disabled";
  } else if (build == "standard") {
    //buildFunction_ = mkfit::runBuildingTestPlexStandard;
    throw cms::Exception("Configuration") << "standard is temporarily disabled";
  } else if (build == "cloneEngine") {
    //buildFunction_ = mkfit::runBuildingTestPlexCloneEngine;
  } else {
    throw cms::Exception("Configuration")
        << "Invalid value for parameter 'buildingRoutine' " << build << ", allowed are bestHit, standard, cloneEngine";
  }

  // TODO: what to do when we have multiple instances of MkFitProducer in a job?
  mkfit::MkBuilderWrapper::populate();
}

void MkFitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelHits", edm::InputTag("mkFitSiPixelHits"));
  desc.add("stripHits", edm::InputTag("mkFitSiStripHits"));
  desc.add("eventOfHits", edm::InputTag("mkFitEventOfHits"));
  desc.add("seeds", edm::InputTag("mkFitSeedConverter"));
  desc.add("clustersToSkip", edm::InputTag());
  desc.add<std::string>("buildingRoutine", "cloneEngine")
      ->setComment("Valid values are: 'bestHit', 'standard', 'cloneEngine'");
  desc.add<edm::ESInputTag>("config")->setComment(
      "ESProduct that has the mkFit configuration parameters for this iteration");
  desc.add("seedCleaning", true)->setComment("Clean seeds within mkFit");
  desc.add("removeDuplicates", true)->setComment("Run duplicate removal within mkFit");
  desc.add("backwardFitInCMSSW", false)
      ->setComment("Do backward fit (to innermost hit) in CMSSW (true) or mkFit (false)");
  desc.addUntracked("mkFitSilent", true)->setComment("Allows to enables printouts from mkFit with 'False'");
  desc.addUntracked("limitConcurrency", false)
      ->setComment(
          "Use tbb::task_arena to limit the internal concurrency to 1; useful only for timing studies when measuring "
          "the module time");

  edm::ParameterSetDescription descCCC;
  descCCC.add<double>("value");
  desc.add("minGoodStripCharge", descCCC);

  descriptions.add("mkFitProducerDefault", desc);
}

std::unique_ptr<mkfit::MkBuilderWrapper> MkFitProducer::beginStream(edm::StreamID iID) const {
  return std::make_unique<mkfit::MkBuilderWrapper>(mkFitSilent_);
}

void MkFitProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& pixelHits = iEvent.get(pixelHitsToken_);
  const auto& stripHits = iEvent.get(stripHitsToken_);
  const auto& eventOfHits = iEvent.get(eventOfHitsToken_);
  const auto& seeds = iEvent.get(seedToken_);
  if (seeds.seeds().empty()) {
    iEvent.emplace(putToken_, mkfit::TrackVec(), not backwardFitInCMSSW_);
    return;
  }
  // This producer does not strictly speaking need the MkFitGeometry,
  // but the ESProducer sets global variables (yes, that "feature"
  // should be removed), so getting the MkFitGeometry makes it
  // sure that the ESProducer is called even if the input/output
  // converters
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);
  const auto& mkFitIterConfig = iSetup.getData(mkFitIterConfigToken_);

  const std::vector<bool>* pixelMaskPtr = nullptr;
  std::vector<bool> pixelMask;
  std::vector<bool> stripMask(stripHits.hits().size(), false);
  if (not pixelMaskToken_.isUninitialized()) {
    if (not pixelHits.hits().empty()) {
      const auto& pixelContainerMask = iEvent.get(pixelMaskToken_);
      pixelMask.resize(pixelContainerMask.size(), false);
      if UNLIKELY (pixelContainerMask.refProd().id() != pixelHits.clustersID()) {
        throw cms::Exception("LogicError") << "MkFitHitWrapper has pixel cluster ID " << pixelHits.clustersID()
                                           << " but pixel cluster mask has " << pixelContainerMask.refProd().id();
      }
      pixelContainerMask.copyMaskTo(pixelMask);
      pixelMaskPtr = &pixelMask;
    }

    if (not stripHits.hits().empty()) {
      const auto& stripContainerMask = iEvent.get(stripMaskToken_);
      if UNLIKELY (stripContainerMask.refProd().id() != stripHits.clustersID()) {
        throw cms::Exception("LogicError") << "MkFitHitWrapper has strip cluster ID " << stripHits.clustersID()
                                           << " but strip cluster mask has " << stripContainerMask.refProd().id();
      }
      stripContainerMask.copyMaskTo(stripMask);
    }
  } else {
    stripClusterChargeCut(iEvent.get(stripClusterChargeToken_), stripMask);
  }

  // seeds need to be mutable because of the possible cleaning
  auto seeds_mutable = seeds.seeds();
  mkfit::TrackVec tracks;

  auto lambda = [&]() {
    mkfit::run_OneIteration(mkFitGeom.trackerInfo(),
                            mkFitIterConfig,
                            eventOfHits.get(),
                            {pixelMaskPtr, &stripMask},
                            streamCache(iID)->get(),
                            seeds_mutable,
                            tracks,
                            seedCleaning_,
                            not backwardFitInCMSSW_,
                            removeDuplicates_);
  };

  if (limitConcurrency_) {
    tbb::task_arena arena(1);
    arena.execute(std::move(lambda));
  } else {
    tbb::this_task_arena::isolate(std::move(lambda));
  }

  iEvent.emplace(putToken_, std::move(tracks), not backwardFitInCMSSW_);
}

void MkFitProducer::stripClusterChargeCut(const std::vector<float>& stripClusterCharge, std::vector<bool>& mask) const {
  if (mask.size() != stripClusterCharge.size()) {
    throw cms::Exception("LogicError") << "Mask size (" << mask.size() << ") inconsistent with number of hits ("
                                       << stripClusterCharge.size() << ")";
  }
  for (int i = 0, end = stripClusterCharge.size(); i < end; ++i) {
    // mask == true means skip the cluster
    mask[i] = mask[i] || (stripClusterCharge[i] <= minGoodStripCharge_);
  }
}

DEFINE_FWK_MODULE(MkFitProducer);

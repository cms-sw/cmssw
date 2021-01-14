#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit includes
#include "ConfigWrapper.h"
#include "Event.h"
#include "LayerNumberConverter.h"
#include "mkFit/buildtestMPlex.h"
#include "mkFit/MkBuilderWrapper.h"

// TBB includes
#include "tbb/task_arena.h"

// std includes
#include <functional>

class MkFitProducer : public edm::global::EDProducer<edm::StreamCache<mkfit::MkBuilderWrapper> > {
public:
  explicit MkFitProducer(edm::ParameterSet const& iConfig);
  ~MkFitProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<mkfit::MkBuilderWrapper> beginStream(edm::StreamID) const override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<MkFitHitWrapper> hitToken_;
  const edm::EDGetTokenT<MkFitSeedWrapper> seedToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  edm::EDPutTokenT<MkFitOutputWrapper> putToken_;
  std::function<double(mkfit::Event&, mkfit::MkBuilder&)> buildFunction_;
  bool backwardFitInCMSSW_;
  bool mkFitSilent_;
};

MkFitProducer::MkFitProducer(edm::ParameterSet const& iConfig)
    : hitToken_{consumes<MkFitHitWrapper>(iConfig.getParameter<edm::InputTag>("hits"))},
      seedToken_{consumes<MkFitSeedWrapper>(iConfig.getParameter<edm::InputTag>("seeds"))},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      putToken_{produces<MkFitOutputWrapper>()},
      backwardFitInCMSSW_{iConfig.getParameter<bool>("backwardFitInCMSSW")},
      mkFitSilent_{iConfig.getUntrackedParameter<bool>("mkFitSilent")} {
  const auto build = iConfig.getParameter<std::string>("buildingRoutine");
  if (build == "bestHit") {
    buildFunction_ = mkfit::runBuildingTestPlexBestHit;
  } else if (build == "standard") {
    buildFunction_ = mkfit::runBuildingTestPlexStandard;
  } else if (build == "cloneEngine") {
    buildFunction_ = mkfit::runBuildingTestPlexCloneEngine;
  } else {
    throw cms::Exception("Configuration") << "Invalid value for parameter 'buildingRoutine' " << build
                                          << ", allowed are bestHit, standard, cloneEngine";
  }

  const auto seedClean = iConfig.getParameter<std::string>("seedCleaning");
  auto seedCleanOpt = mkfit::ConfigWrapper::SeedCleaningOpts::noCleaning;
  if (seedClean == "none") {
    seedCleanOpt = mkfit::ConfigWrapper::SeedCleaningOpts::noCleaning;
  } else if (seedClean == "N2") {
    seedCleanOpt = mkfit::ConfigWrapper::SeedCleaningOpts::cleanSeedsN2;
  } else {
    throw cms::Exception("Configuration")
        << "Invalida value for parameter 'seedCleaning' " << seedClean << ", allowed are none, N2";
  }

  auto backwardFitOpt =
      backwardFitInCMSSW_ ? mkfit::ConfigWrapper::BackwardFit::noFit : mkfit::ConfigWrapper::BackwardFit::toFirstLayer;

  // TODO: what to do when we have multiple instances of MkFitProducer in a job?
  mkfit::MkBuilderWrapper::populate();
  mkfit::ConfigWrapper::initializeForCMSSW(seedCleanOpt, backwardFitOpt, mkFitSilent_);
  mkfit::ConfigWrapper::setRemoveDuplicates(iConfig.getParameter<bool>("removeDuplicates"));
}

void MkFitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("hits", edm::InputTag("mkFitHitConverter"));
  desc.add("seeds", edm::InputTag("mkFitSeedConverter"));
  desc.add<std::string>("buildingRoutine", "cloneEngine")
      ->setComment("Valid values are: 'bestHit', 'standard', 'cloneEngine'");
  desc.add<std::string>("seedCleaning", "N2")->setComment("Valid values are: 'none', 'N2'");
  desc.add("removeDuplicates", true)->setComment("Run duplicate removal within mkFit");
  desc.add("backwardFitInCMSSW", false)
      ->setComment("Do backward fit (to innermost hit) in CMSSW (true) or mkFit (false)");
  desc.addUntracked("mkFitSilent", true)->setComment("Allows to enables printouts from mkFit with 'False'");

  descriptions.add("mkFitProducer", desc);
}

std::unique_ptr<mkfit::MkBuilderWrapper> MkFitProducer::beginStream(edm::StreamID iID) const {
  return std::make_unique<mkfit::MkBuilderWrapper>();
}

namespace {
  std::once_flag geometryFlag;
}
void MkFitProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& hits = iEvent.get(hitToken_);
  const auto& seeds = iEvent.get(seedToken_);
  // This producer does not strictly speaking need the MkFitGeometry,
  // but the ESProducer sets global variables (yes, that "feature"
  // should be removed), so getting the MkFitGeometry makes it
  // sure that the ESProducer is called even if the input/output
  // converters
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  // Initialize the number of layers, has to be done exactly once in
  // the whole program.
  // TODO: the mechanism needs to be improved...
  std::call_once(geometryFlag, [nlayers = mkFitGeom.layerNumberConverter().nLayers()]() {
    mkfit::ConfigWrapper::setNTotalLayers(nlayers);
  });

  // CMSSW event ID (64-bit unsigned) does not fit in int
  // In addition, unique ID requires also lumi and run
  // But does the event ID really matter within mkFit?
  mkfit::Event ev(iEvent.id().event());

  ev.setInputFromCMSSW(hits.hits(), seeds.seeds());

  tbb::this_task_arena::isolate([&]() { buildFunction_(ev, streamCache(iID)->get()); });

  iEvent.emplace(putToken_, std::move(ev.candidateTracks_), std::move(ev.fitTracks_));
}

DEFINE_FWK_MODULE(MkFitProducer);

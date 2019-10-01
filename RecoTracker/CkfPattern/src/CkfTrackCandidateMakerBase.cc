#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include <FWCore/Utilities/interface/ESInputTag.h>

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"

#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerBySharedInput.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"

#include <algorithm>
#include <functional>

// #define VI_SORTSEED
// #define VI_REPRODUCIBLE
// #define VI_TBB

#include <thread>
#ifdef VI_TBB
#include "tbb/parallel_for.h"
#endif

#include "RecoTracker/CkfPattern/interface/PrintoutHelper.h"

using namespace edm;
using namespace std;

namespace {
  std::unique_ptr<BaseCkfTrajectoryBuilder> createBaseCkfTrajectoryBuilder(const edm::ParameterSet& pset,
                                                                           edm::ConsumesCollector& iC) {
    return BaseCkfTrajectoryBuilderFactory::get()->create(pset.getParameter<std::string>("ComponentType"), pset, iC);
  }
}  // namespace

namespace cms {
  CkfTrackCandidateMakerBase::CkfTrackCandidateMakerBase(edm::ParameterSet const& conf, edm::ConsumesCollector&& iC)
      : theTrackCandidateOutput(true),
        theTrajectoryOutput(false),
        useSplitting(conf.getParameter<bool>("useHitsSplitting")),
        doSeedingRegionRebuilding(conf.getParameter<bool>("doSeedingRegionRebuilding")),
        cleanTrajectoryAfterInOut(conf.getParameter<bool>("cleanTrajectoryAfterInOut")),
        reverseTrajectories(conf.existsAs<bool>("reverseTrajectories") &&
                            conf.getParameter<bool>("reverseTrajectories")),
        theMaxNSeeds(conf.getParameter<unsigned int>("maxNSeeds")),
        theTrajectoryBuilder(
            createBaseCkfTrajectoryBuilder(conf.getParameter<edm::ParameterSet>("TrajectoryBuilderPSet"), iC)),
        theTrajectoryCleanerName(conf.getParameter<std::string>("TrajectoryCleaner")),
        theTrajectoryCleaner(nullptr),
        theInitialState(std::make_unique<TransientInitialStateEstimator>(
            conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters"))),
        theMagFieldName(conf.exists("SimpleMagneticField") ? conf.getParameter<std::string>("SimpleMagneticField")
                                                           : ""),
        theNavigationSchoolName(conf.getParameter<std::string>("NavigationSchool")),
        theNavigationSchool(nullptr),
        maxSeedsBeforeCleaning_(0),
        theMTELabel(iC.consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
        skipClusters_(false),
        phase2skipClusters_(false) {
    theSeedLabel = iC.consumes<edm::View<TrajectorySeed>>(conf.getParameter<edm::InputTag>("src"));
#ifndef VI_REPRODUCIBLE
    if (conf.exists("maxSeedsBeforeCleaning"))
      maxSeedsBeforeCleaning_ = conf.getParameter<unsigned int>("maxSeedsBeforeCleaning");
#endif
    if (conf.existsAs<edm::InputTag>("clustersToSkip")) {
      skipClusters_ = true;
      maskPixels_ = iC.consumes<PixelClusterMask>(conf.getParameter<edm::InputTag>("clustersToSkip"));
      maskStrips_ = iC.consumes<StripClusterMask>(conf.getParameter<edm::InputTag>("clustersToSkip"));
    }
    //FIXME:: just temporary solution for phase2!
    if (conf.existsAs<edm::InputTag>("phase2clustersToSkip")) {
      phase2skipClusters_ = true;
      maskPixels_ = iC.consumes<PixelClusterMask>(conf.getParameter<edm::InputTag>("phase2clustersToSkip"));
      maskPhase2OTs_ = iC.consumes<Phase2OTClusterMask>(conf.getParameter<edm::InputTag>("phase2clustersToSkip"));
    }
#ifndef VI_REPRODUCIBLE
    std::string cleaner = conf.getParameter<std::string>("RedundantSeedCleaner");
    if (cleaner == "CachingSeedCleanerBySharedInput") {
      int numHitsForSeedCleaner =
          conf.existsAs<int>("numHitsForSeedCleaner") ? conf.getParameter<int>("numHitsForSeedCleaner") : 4;
      int onlyPixelHits = conf.existsAs<bool>("onlyPixelHitsForSeedCleaner")
                              ? conf.getParameter<bool>("onlyPixelHitsForSeedCleaner")
                              : false;
      theSeedCleaner = std::make_unique<CachingSeedCleanerBySharedInput>(numHitsForSeedCleaner, onlyPixelHits);
    } else if (cleaner != "none") {
      throw cms::Exception("RedundantSeedCleaner not found, please use CachingSeedCleanerBySharedInput ro none",
                           cleaner);
    }
#endif

#ifdef VI_REPRODUCIBLE
    std::cout << "CkfTrackCandidateMaker in reproducible setting" << std::endl;
    assert(nullptr == theSeedCleaner);
    assert(0 >= maxSeedsBeforeCleaning_);
#endif
  }

  // Virtual destructor needed.
  CkfTrackCandidateMakerBase::~CkfTrackCandidateMakerBase() noexcept(false) {}

  void CkfTrackCandidateMakerBase::beginRunBase(edm::Run const& r, EventSetup const& es) { /* no op*/
  }

  void CkfTrackCandidateMakerBase::setEventSetup(const edm::EventSetup& es) {
    //services
    es.get<TrackerRecoGeometryRecord>().get(theGeomSearchTracker);
    es.get<IdealMagneticFieldRecord>().get(theMagFieldName, theMagField);
    //    edm::ESInputTag mfESInputTag(mfName);
    //    es.get<IdealMagneticFieldRecord>().get(mfESInputTag,theMagField );

    edm::ESHandle<TrajectoryCleaner> trajectoryCleanerH;
    es.get<TrajectoryCleaner::Record>().get(theTrajectoryCleanerName, trajectoryCleanerH);
    theTrajectoryCleaner = trajectoryCleanerH.product();

    edm::ESHandle<NavigationSchool> navigationSchoolH;
    es.get<NavigationSchoolRecord>().get(theNavigationSchoolName, navigationSchoolH);
    theNavigationSchool = navigationSchoolH.product();
    theTrajectoryBuilder->setNavigationSchool(theNavigationSchool);
  }

  // Functions that gets called by framework every event
  void CkfTrackCandidateMakerBase::produceBase(edm::Event& e, const edm::EventSetup& es) {
    // getting objects from the EventSetup
    setEventSetup(es);

    // set the correct navigation
    // NavigationSetter setter( *theNavigationSchool);

    // propagator
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get("AnyDirectionAnalyticalPropagator", thePropagator);

    // method for Debugging
    printHitsDebugger(e);

    // Step A: set Event for the TrajectoryBuilder
    edm::Handle<MeasurementTrackerEvent> data;
    e.getByToken(theMTELabel, data);

    std::unique_ptr<MeasurementTrackerEvent> dataWithMasks;
    if (skipClusters_) {
      edm::Handle<PixelClusterMask> pixelMask;
      e.getByToken(maskPixels_, pixelMask);
      edm::Handle<StripClusterMask> stripMask;
      e.getByToken(maskStrips_, stripMask);
      dataWithMasks = std::make_unique<MeasurementTrackerEvent>(*data, *stripMask, *pixelMask);
      //std::cout << "Trajectory builder " << conf_.getParameter<std::string>("@module_label") << " created with masks " << std::endl;
      theTrajectoryBuilder->setEvent(e, es, &*dataWithMasks);
    } else if (phase2skipClusters_) {
      //FIXME:just temporary solution for phase2!
      edm::Handle<PixelClusterMask> pixelMask;
      e.getByToken(maskPixels_, pixelMask);
      edm::Handle<Phase2OTClusterMask> phase2OTMask;
      e.getByToken(maskPhase2OTs_, phase2OTMask);
      dataWithMasks = std::make_unique<MeasurementTrackerEvent>(*data, *pixelMask, *phase2OTMask);
      //std::cout << "Trajectory builder " << conf_.getParameter<std::string>("@module_label") << " created with phase2 masks " << std::endl;
      theTrajectoryBuilder->setEvent(e, es, &*dataWithMasks);
    } else {
      //std::cout << "Trajectory builder " << conf_.getParameter<std::string>("@module_label") << " created without masks " << std::endl;
      theTrajectoryBuilder->setEvent(e, es, &*data);
    }
    // TISE ES must be set here due to dependence on theTrajectoryBuilder
    theInitialState->setEventSetup(
        es, static_cast<TkTransientTrackingRecHitBuilder const*>(theTrajectoryBuilder->hitBuilder())->cloner());

    // Step B: Retrieve seeds

    edm::Handle<View<TrajectorySeed>> collseed;
    e.getByToken(theSeedLabel, collseed);

    // Step C: Create empty output collection
    auto output = std::make_unique<TrackCandidateCollection>();
    auto outputT = std::make_unique<std::vector<Trajectory>>();
    auto outputSeedStopInfos = std::make_unique<std::vector<SeedStopInfo>>(collseed->size());

    if ((*collseed).size() > theMaxNSeeds) {
      LogError("TooManySeeds") << "Exceeded maximum numeber of seeds! theMaxNSeeds=" << theMaxNSeeds
                               << " nSeed=" << (*collseed).size();
      if (theTrackCandidateOutput) {
        e.put(std::move(output));
      }
      if (theTrajectoryOutput) {
        e.put(std::move(outputT));
      }
      e.put(std::move(outputSeedStopInfos));
      return;
    }

    // Step D: Invoke the building algorithm
    if (!(*collseed).empty()) {
      unsigned int lastCleanResult = 0;
      std::vector<Trajectory> rawResult;
      rawResult.reserve(collseed->size() * 4);

      if (theSeedCleaner)
        theSeedCleaner->init(&rawResult);

      // method for debugging
      countSeedsDebugger();

      // the mutex
      std::mutex theMutex;
      using Lock = std::unique_lock<std::mutex>;

      // Loop over seeds
      size_t collseed_size = collseed->size();

      unsigned int indeces[collseed_size];
      for (auto i = 0U; i < collseed_size; ++i)
        indeces[i] = i;

#ifdef VI_SORTSEED
      // std::random_shuffle(indeces,indeces+collseed_size);

      // here only for reference: does not seems to help

      auto const& seeds = *collseed;

      float val[collseed_size];
      for (auto i = 0U; i < collseed_size; ++i) {
        val[i] = seeds[i].startingState().pt();
      };
      //  { val[i] =  std::abs((*seeds[i].recHits().first).surface()->eta());}

      /*
      unsigned long long val[collseed_size];
      for (auto i=0U; i< collseed_size; ++i) {
        if (seeds[i].nHits()<2) { val[i]=0; continue;}
        auto h = seeds[i].recHits().first;
        auto const & hit = static_cast<BaseTrackerRecHit const&>(*h);
        val[i] = hit.firstClusterRef().key();
        if (++h != seeds[i].recHits().second) {
          auto	const &	hit = static_cast<BaseTrackerRecHit const&>(*h);
    	  val[i] |= (unsigned long long)(hit.firstClusterRef().key())<<32;
        }
      }
      */
      std::sort(indeces, indeces + collseed_size, [&](unsigned int i, unsigned int j) { return val[i] < val[j]; });

      // std::cout << spt(indeces[0]) << ' ' << spt(indeces[collseed_size-1]) << std::endl;
#endif

      std::atomic<unsigned int> ntseed(0);
      auto theLoop = [&](size_t ii) {
        auto j = indeces[ii];

        ntseed++;

        // to be moved inside a par section (how with tbb??)
        std::vector<Trajectory> theTmpTrajectories;

        LogDebug("CkfPattern") << "======== Begin to look for trajectories from seed " << j << " ========\n";

        {
          Lock lock(theMutex);
          // Check if seed hits already used by another track
          if (theSeedCleaner && !theSeedCleaner->good(&((*collseed)[j]))) {
            LogDebug("CkfTrackCandidateMakerBase") << " Seed cleaning kills seed " << j;
            (*outputSeedStopInfos)[j].setStopReason(SeedStopReason::SEED_CLEANING);
            return;  // from the lambda!
          }
        }

        // Build trajectory from seed outwards
        theTmpTrajectories.clear();
        unsigned int nCandPerSeed = 0;
        auto const& startTraj =
            theTrajectoryBuilder->buildTrajectories((*collseed)[j], theTmpTrajectories, nCandPerSeed, nullptr);
        {
          Lock lock(theMutex);
          (*outputSeedStopInfos)[j].setCandidatesPerSeed(nCandPerSeed);
          if (theTmpTrajectories.empty()) {
            (*outputSeedStopInfos)[j].setStopReason(SeedStopReason::NO_TRAJECTORY);
            return;  // from the lambda!
          }
        }

        LogDebug("CkfPattern") << "======== In-out trajectory building found " << theTmpTrajectories.size()
                               << " trajectories from seed " << j << " ========\n"
                               << PrintoutHelper::dumpCandidates(theTmpTrajectories);

        if (cleanTrajectoryAfterInOut) {
          // Select the best trajectory from this seed (declare others invalid)
          theTrajectoryCleaner->clean(theTmpTrajectories);

          LogDebug("CkfPattern") << "======== In-out trajectory cleaning gave the following "
                                 << theTmpTrajectories.size() << " valid trajectories from seed " << j << " ========\n"
                                 << PrintoutHelper::dumpCandidates(theTmpTrajectories);
        }

        // Optionally continue building trajectory back through
        // seed and if possible further inwards.

        if (doSeedingRegionRebuilding) {
          theTrajectoryBuilder->rebuildTrajectories(startTraj, (*collseed)[j], theTmpTrajectories);

          LogDebug("CkfPattern") << "======== Out-in trajectory building found " << theTmpTrajectories.size()
                                 << " valid/invalid trajectories from seed " << j << " ========\n"
                                 << PrintoutHelper::dumpCandidates(theTmpTrajectories);
          if (theTmpTrajectories.empty()) {
            Lock lock(theMutex);
            (*outputSeedStopInfos)[j].setStopReason(SeedStopReason::SEED_REGION_REBUILD);
            return;
          }
        }

        // Select the best trajectory from this seed (after seed region rebuilding, can be more than one)
        theTrajectoryCleaner->clean(theTmpTrajectories);

        LogDebug("CkfPattern") << "======== Trajectory cleaning gave the following " << theTmpTrajectories.size()
                               << " valid trajectories from seed " << j << " ========\n"
                               << PrintoutHelper::dumpCandidates(theTmpTrajectories);

        {
          Lock lock(theMutex);
          for (vector<Trajectory>::iterator it = theTmpTrajectories.begin(); it != theTmpTrajectories.end(); it++) {
            if (it->isValid()) {
              it->setSeedRef(collseed->refAt(j));
              (*outputSeedStopInfos)[j].setStopReason(SeedStopReason::NOT_STOPPED);
              // Store trajectory
              rawResult.push_back(std::move(*it));
              // Tell seed cleaner which hits this trajectory used.
              //TO BE FIXED: this cut should be configurable via cfi file
              if (theSeedCleaner && rawResult.back().foundHits() > 3)
                theSeedCleaner->add(&rawResult.back());
              //if (theSeedCleaner ) theSeedCleaner->add( & (*it) );
            }
          }
        }

        theTmpTrajectories.clear();

        LogDebug("CkfPattern") << "rawResult trajectories found so far = " << rawResult.size();

        {
          Lock lock(theMutex);
          if (maxSeedsBeforeCleaning_ > 0 && rawResult.size() > maxSeedsBeforeCleaning_ + lastCleanResult) {
            theTrajectoryCleaner->clean(rawResult);
            rawResult.erase(
                std::remove_if(rawResult.begin() + lastCleanResult, rawResult.end(), std::not_fn(&Trajectory::isValid)),
                rawResult.end());
            lastCleanResult = rawResult.size();
          }
        }
      };
      // end of loop over seeds

#ifdef VI_TBB
      tbb::parallel_for(0UL, collseed_size, 1UL, theLoop);
#else
#ifdef VI_OMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
      for (size_t j = 0; j < collseed_size; j++) {
        theLoop(j);
      }
#endif
      assert(ntseed == collseed_size);
      if (theSeedCleaner)
        theSeedCleaner->done();

        // std::cout << "VICkfPattern " << "rawResult trajectories found = " << rawResult.size() << " in " << ntseed << " seeds " << collseed_size << std::endl;

#ifdef VI_REPRODUCIBLE
      // sort trajectory
      std::sort(rawResult.begin(), rawResult.end(), [](const Trajectory& a, const Trajectory& b) {
        return a.seedRef().key() < b.seedRef().key();
      });
      //{ return a.chiSquared()*b.ndof() < b.chiSquared()*a.ndof();});
#endif

      // Step E: Clean the results to avoid duplicate tracks
      // Rejected ones just flagged as invalid.
      theTrajectoryCleaner->clean(rawResult);

      LogDebug("CkfPattern") << "======== Final cleaning of entire event found " << rawResult.size()
                             << " valid/invalid trajectories =======" << endl
                             << PrintoutHelper::dumpCandidates(rawResult);

      LogDebug("CkfPattern") << "removing invalid trajectories.";

      // Assuming here that theLoop() gives at most one Trajectory per seed
      for (const auto& traj : rawResult) {
        if (!traj.isValid()) {
          const auto seedIndex = traj.seedRef().key();
          if ((*outputSeedStopInfos)[seedIndex].stopReason() == SeedStopReason::NOT_STOPPED) {
            (*outputSeedStopInfos)[seedIndex].setStopReason(SeedStopReason::FINAL_CLEAN);
          }
        }
      }

      vector<Trajectory>& unsmoothedResult(rawResult);
      unsmoothedResult.erase(
          std::remove_if(unsmoothedResult.begin(), unsmoothedResult.end(), std::not_fn(&Trajectory::isValid)),
          unsmoothedResult.end());
      unsmoothedResult.shrink_to_fit();
      // If requested, reverse the trajectories creating a new 1-hit seed on the last measurement of the track
      if (reverseTrajectories) {
        for (auto it = unsmoothedResult.begin(), ed = unsmoothedResult.end(); it != ed; ++it) {
          // reverse the trajectory only if it has valid hit on the last measurement (should happen)
          if (it->lastMeasurement().updatedState().isValid() && it->lastMeasurement().recHit().get() != nullptr &&
              it->lastMeasurement().recHit()->isValid()) {
            // I can't use reverse in place, because I want to change the seed
            // 1) reverse propagation direction
            PropagationDirection direction = it->direction();
            if (direction == alongMomentum)
              direction = oppositeToMomentum;
            else if (direction == oppositeToMomentum)
              direction = alongMomentum;
            // 2) make a seed
            TrajectoryStateOnSurface const& initState = it->lastMeasurement().updatedState();
            auto initId = it->lastMeasurement().recHitR().rawId();
            PTrajectoryStateOnDet&& state = trajectoryStateTransform::persistentState(initState, initId);
            TrajectorySeed::recHitContainer hits;
            hits.push_back(it->lastMeasurement().recHit()->hit()->clone());
            std::shared_ptr<const TrajectorySeed> seed(new TrajectorySeed(state, std::move(hits), direction));
            // 3) make a trajectory
            Trajectory trajectory(seed, direction);
            trajectory.setNLoops(it->nLoops());
            trajectory.setSeedRef(it->seedRef());
            trajectory.setStopReason(it->stopReason());
            // 4) push states in reversed order
            Trajectory::DataContainer& meas = it->measurements();
            trajectory.reserve(meas.size());
            for (auto itmeas = meas.rbegin(), endmeas = meas.rend(); itmeas != endmeas; ++itmeas) {
              trajectory.push(std::move(*itmeas));
            }
            // replace
            (*it) = std::move(trajectory);
          } else {
            edm::LogWarning("CkfPattern_InvalidLastMeasurement")
                << "Last measurement of the trajectory is invalid, cannot reverse it";
          }
        }
      }

      int viTotHits = 0;

      if (theTrackCandidateOutput) {
        // Step F: Convert to TrackCandidates
        output->reserve(unsmoothedResult.size());
        Traj2TrackHits t2t(theTrajectoryBuilder->hitBuilder(), true);

        for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin(); it != unsmoothedResult.end(); ++it) {
          LogDebug("CkfPattern") << "copying " << (useSplitting ? "splitted" : "un-splitted")
                                 << " hits from trajectory";
          edm::OwnVector<TrackingRecHit> recHits;
          if (it->direction() != alongMomentum)
            LogDebug("CkfPattern") << "not along momentum... " << std::endl;
          t2t(*it, recHits, useSplitting);

          viTotHits += recHits.size();

          LogDebug("CkfPattern") << "getting initial state.";
          Trajectory trialTrajectory = (*it);
          std::pair<TrajectoryStateOnSurface, const GeomDet*> initState;
          bool failed = false;

          do {
            // Drop last hit if previous backFitter was not successful
            if (failed) {
              LogDebug("CkfPattern") << "removing last hit";
              trialTrajectory.pop();
              LogDebug("CkfPattern") << "hits remaining " << trialTrajectory.foundHits();
            }

            // Get inner state
            const bool doBackFit = (!doSeedingRegionRebuilding) & (!reverseTrajectories);
            initState = theInitialState->innerState(trialTrajectory, doBackFit);

            // Check if that was successful
            failed = (!initState.first.isValid()) || initState.second == nullptr ||
                     edm::isNotFinite(initState.first.globalPosition().x());
          } while (failed && trialTrajectory.foundHits() > 3);

          if (failed) {
            const auto seedIndex = it->seedRef().key();
            (*outputSeedStopInfos)[seedIndex].setStopReason(SeedStopReason::SMOOTHING_FAILED);
            continue;
          }

          PTrajectoryStateOnDet state;
          if (useSplitting && (initState.second != recHits.front().det()) && recHits.front().det()) {
            LogDebug("CkfPattern") << "propagating to hit front in case of splitting.";
            TrajectoryStateOnSurface&& propagated =
                thePropagator->propagate(initState.first, recHits.front().det()->surface());
            if (!propagated.isValid())
              continue;
            state = trajectoryStateTransform::persistentState(propagated, recHits.front().rawId());
          } else
            state =
                trajectoryStateTransform::persistentState(initState.first, initState.second->geographicalId().rawId());
          LogDebug("CkfPattern") << "pushing a TrackCandidate.";
          output->emplace_back(recHits, it->seed(), state, it->seedRef(), it->nLoops(), (uint8_t)it->stopReason());
        }
      }  //output trackcandidates

      edm::ESHandle<TrackerGeometry> tracker;
      es.get<TrackerDigiGeometryRecord>().get(tracker);
      LogTrace("CkfPattern|TrackingRegressionTest") << "========== CkfTrackCandidateMaker Info =========="
                                                    << "number of Seed: " << collseed->size() << '\n'
                                                    << PrintoutHelper::regressionTest(*tracker, unsmoothedResult);

      assert(viTotHits >= 0);  // just to use it...
      // std::cout << "VICkfPattern result " << output->size() << " " << viTotHits << std::endl;

      if (theTrajectoryOutput) {
        outputT->swap(unsmoothedResult);
      }

    }  // end of ((*collseed).size()>0)

    // method for debugging
    deleteAssocDebugger();

    // Step G: write output to file
    if (theTrackCandidateOutput) {
      e.put(std::move(output));
    }
    if (theTrajectoryOutput) {
      e.put(std::move(outputT));
    }
    e.put(std::move(outputSeedStopInfos));
  }

}  // namespace cms

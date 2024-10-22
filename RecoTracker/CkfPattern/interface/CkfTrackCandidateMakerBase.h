#ifndef CkfTrackCandidateMakerBase_h
#define CkfTrackCandidateMakerBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include <memory>

class TransientInitialStateEstimator;
class NavigationSchoolRecord;
class TrackerDigiGeometryRecord;

namespace cms {
  class CkfTrackCandidateMakerBase {
  public:
    explicit CkfTrackCandidateMakerBase(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

    virtual ~CkfTrackCandidateMakerBase() noexcept(false);

    virtual void beginRunBase(edm::Run const&, edm::EventSetup const& es);

    virtual void produceBase(edm::Event& e, const edm::EventSetup& es);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

  protected:
    bool theTrackCandidateOutput;
    bool theTrajectoryOutput;
    bool useSplitting;
    bool doSeedingRegionRebuilding;
    bool cleanTrajectoryAfterInOut;
    bool reverseTrajectories;
    bool produceSeedStopReasons_;

    unsigned int theMaxNSeeds;

    std::unique_ptr<BaseCkfTrajectoryBuilder> theTrajectoryBuilder;

    edm::ESGetToken<TrajectoryCleaner, TrajectoryCleaner::Record> theTrajectoryCleanerToken;
    const TrajectoryCleaner* theTrajectoryCleaner;

    std::unique_ptr<TransientInitialStateEstimator> theInitialState;

    edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> theNavigationSchoolToken;
    const NavigationSchool* theNavigationSchool;

    edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> theTrackerToken;

    std::unique_ptr<RedundantSeedCleaner> theSeedCleaner;

    unsigned int maxSeedsBeforeCleaning_;

    edm::EDGetTokenT<edm::View<TrajectorySeed> > theSeedLabel;
    edm::EDGetTokenT<MeasurementTrackerEvent> theMTELabel;

    edm::InputTag const clustersToSkipTag_;
    bool const skipClusters_;

    edm::InputTag const phase2ClustersToSkipTag_;
    bool const skipPhase2Clusters_;

    typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelClusterMask;
    typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > StripClusterMask;
    typedef edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > Phase2OTClusterMask;
    edm::EDGetTokenT<PixelClusterMask> maskPixels_;
    edm::EDGetTokenT<StripClusterMask> maskStrips_;
    edm::EDGetTokenT<Phase2OTClusterMask> maskPhase2OTs_;

    // methods for debugging
    virtual TrajectorySeedCollection::const_iterator lastSeed(TrajectorySeedCollection const& theSeedColl) {
      return theSeedColl.end();
    }
    virtual void printHitsDebugger(edm::Event& e) { ; }
    virtual void countSeedsDebugger() { ; }
    virtual void deleteAssocDebugger() { ; }

  private:
    /// Initialize EventSetup objects at each event
    void setEventSetup(const edm::EventSetup& es);
  };
}  // namespace cms

#endif

#ifndef CkfTrackCandidateMakerBase_h
#define CkfTrackCandidateMakerBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

namespace cms {
  class CkfTrackCandidateMakerBase {
  public:
    explicit CkfTrackCandidateMakerBase(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

    virtual ~CkfTrackCandidateMakerBase() noexcept(false);

    virtual void beginRunBase(edm::Run const&, edm::EventSetup const& es);

    virtual void produceBase(edm::Event& e, const edm::EventSetup& es);

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

    std::string theTrajectoryCleanerName;
    const TrajectoryCleaner* theTrajectoryCleaner;

    std::unique_ptr<TransientInitialStateEstimator> theInitialState;

    const std::string theMagFieldName;
    edm::ESHandle<MagneticField> theMagField;
    edm::ESHandle<GeometricSearchTracker> theGeomSearchTracker;

    std::string theNavigationSchoolName;
    const NavigationSchool* theNavigationSchool;

    std::unique_ptr<RedundantSeedCleaner> theSeedCleaner;

    unsigned int maxSeedsBeforeCleaning_;

    edm::EDGetTokenT<edm::View<TrajectorySeed> > theSeedLabel;
    edm::EDGetTokenT<MeasurementTrackerEvent> theMTELabel;

    bool skipClusters_;
    bool phase2skipClusters_;
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

#ifndef CkfTrackCandidateMakerBase_h
#define CkfTrackCandidateMakerBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class TransientInitialStateEstimator;

namespace cms
{
  class CkfTrackCandidateMakerBase  {
  public:

    explicit CkfTrackCandidateMakerBase(const edm::ParameterSet& conf);

    virtual ~CkfTrackCandidateMakerBase();

    virtual void beginRunBase (edm::Run const & , edm::EventSetup const & es);

    virtual void produceBase(edm::Event& e, const edm::EventSetup& es);

  protected:

    edm::ParameterSet conf_;

    bool theTrackCandidateOutput;
    bool theTrajectoryOutput;
    bool useSplitting;
    bool doSeedingRegionRebuilding;
    bool cleanTrajectoryAfterInOut;
    bool reverseTrajectories;

    unsigned int theMaxNSeeds;

    std::string theTrajectoryBuilderName;
    const BaseCkfTrajectoryBuilder*  theTrajectoryBuilder;

    std::string theTrajectoryCleanerName;
    const TrajectoryCleaner*               theTrajectoryCleaner;

    TransientInitialStateEstimator*  theInitialState;
    
    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

    std::string theNavigationSchoolName;
    const NavigationSchool*       theNavigationSchool;
    
    RedundantSeedCleaner*  theSeedCleaner;
    
    edm::InputTag theSeedLabel;

    unsigned int maxSeedsBeforeCleaning_;
    // methods for debugging
    virtual TrajectorySeedCollection::const_iterator lastSeed(TrajectorySeedCollection const& theSeedColl){return theSeedColl.end();}
    virtual void printHitsDebugger(edm::Event& e){;}
    virtual void countSeedsDebugger(){;}
    virtual void deleteAssocDebugger(){;}
  private:
    /// Initialize EventSetup objects at each event
    void setEventSetup( const edm::EventSetup& es ) ; 
  };
}

#endif

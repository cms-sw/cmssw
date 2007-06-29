#ifndef CkfTrackCandidateMakerBase_h
#define CkfTrackCandidateMakerBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

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

    virtual void beginJobBase (edm::EventSetup const & es);

    virtual void produceBase(edm::Event& e, const edm::EventSetup& es);

  protected:
    edm::ParameterSet conf_;
    const TrackerTrajectoryBuilder*  theTrajectoryBuilder;
    TrajectoryCleaner*               theTrajectoryCleaner;
    TransientInitialStateEstimator*  theInitialState;
    
    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

    const NavigationSchool*       theNavigationSchool;
    
    RedundantSeedCleaner*  theSeedCleaner;

    // methods for debugging
    virtual TrajectorySeedCollection::const_iterator lastSeed(TrajectorySeedCollection& theSeedColl){return theSeedColl.end();}
    virtual void printHitsDebugger(edm::Event& e){;}
    virtual void countSeedsDebugger(){;}
    virtual void deleteAssocDebugger(){;}

  };
}

#endif

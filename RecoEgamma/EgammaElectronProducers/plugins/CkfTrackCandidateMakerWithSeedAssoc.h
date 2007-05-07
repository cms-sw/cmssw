#ifndef CkfTrackCandidateMakerWithSeedAssoc_h
#define CkfTrackCandidateMakerWithSeedAssoc_h

//
// Package:    RecoTracker/CkfTracker
// Class:      CkfTrackCandidateMakeWithSeedAssoc
// 
//
// Description: Produce TrackCandidates from seeds
// and write Associationmap trackcandidates-seeds at the same time
//
//
// Original Author:  Ursula Berthon, Claude Charlot
// close adaptation from CkfTrackCandidateMaker, just adding associationmap
//         Created:  Fri Nov  10 10:29:31 CET 2006
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"

class TransientInitialStateEstimator;

namespace cms
{
  class CkfTrackCandidateMakerWithSeedAssoc : public edm::EDProducer
  {
  public:

    explicit CkfTrackCandidateMakerWithSeedAssoc(const edm::ParameterSet& conf);

    virtual ~CkfTrackCandidateMakerWithSeedAssoc();

    virtual void beginJob (edm::EventSetup const & es);

    virtual void produce(edm::Event& e, const edm::EventSetup& es);

  private:
    edm::ParameterSet conf_;
    const TrackerTrajectoryBuilder*  theTrajectoryBuilder;
    TrajectoryCleaner*               theTrajectoryCleaner;
    TransientInitialStateEstimator*  theInitialState;
    
    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

    const NavigationSchool*       theNavigationSchool;
    
    RedundantSeedCleaner*  theSeedCleaner;
  };
}

#endif

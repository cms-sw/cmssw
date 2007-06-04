#ifndef CkfTrackCandidateMaker_h
#define CkfTrackCandidateMaker_h

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

class TransientInitialStateEstimator;

namespace cms
{
  class CkfTrackCandidateMaker : public edm::EDProducer
  {
  public:

    explicit CkfTrackCandidateMaker(const edm::ParameterSet& conf);

    virtual ~CkfTrackCandidateMaker();

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
  };
}

#endif

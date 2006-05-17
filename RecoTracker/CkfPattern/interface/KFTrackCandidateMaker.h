#ifndef KFTrackCandidateMaker_h
#define KFTrackCandidateMaker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/CkfPattern/interface/CombinatorialTrajectoryBuilder.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

class TransientInitialStateEstimator;

namespace cms
{
  class KFTrackCandidateMaker : public edm::EDProducer
  {
  public:

    explicit KFTrackCandidateMaker(const edm::ParameterSet& conf);

    virtual ~KFTrackCandidateMaker();

    virtual void produce(edm::Event& e, const edm::EventSetup& es);

  private:
    edm::ParameterSet conf_;
    CombinatorialTrajectoryBuilder*  theCombinatorialTrajectoryBuilder;
    TrajectoryCleaner*               theTrajectoryCleaner;
    TransientInitialStateEstimator*  theInitialState;

    edm::ESHandle<MagneticField>                theMagField;
    edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker;

    const MeasurementTracker*     theMeasurementTracker;
    const NavigationSchool*       theNavigationSchool;

    bool isInitialized;
  };
}

#endif

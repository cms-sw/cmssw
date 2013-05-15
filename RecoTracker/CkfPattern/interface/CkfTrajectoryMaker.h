#ifndef CkfTrajectoryMaker_h
#define CkfTrajectoryMaker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 

class TransientInitialStateEstimator;

namespace cms
{
  class CkfTrajectoryMaker : public CkfTrackCandidateMakerBase, public edm::EDProducer
  {
  public:
    typedef std::vector<Trajectory> TrajectoryCollection;

    explicit CkfTrajectoryMaker(const edm::ParameterSet& conf):
      CkfTrackCandidateMakerBase(conf)
    {
      theTrackCandidateOutput=conf.getParameter<bool>("trackCandidateAlso");
      theTrajectoryOutput=true;
      if (theTrackCandidateOutput)
	produces<TrackCandidateCollection>();
      produces<TrajectoryCollection>();
    }

    virtual ~CkfTrajectoryMaker(){;}

    virtual void beginRun (edm::Run const & run, edm::EventSetup const & es) override {beginRunBase(run,es);}

    virtual void produce(edm::Event& e, const edm::EventSetup& es) override {produceBase(e,es);}
  };
}

#endif

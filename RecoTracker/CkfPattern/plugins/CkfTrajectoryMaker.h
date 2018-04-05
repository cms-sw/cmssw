#ifndef CkfTrajectoryMaker_h
#define CkfTrajectoryMaker_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"

class TransientInitialStateEstimator;

namespace cms
{
  class dso_internal CkfTrajectoryMaker : public edm::stream::EDProducer<>, public CkfTrackCandidateMakerBase
  {
  public:
    typedef std::vector<Trajectory> TrajectoryCollection;

    explicit CkfTrajectoryMaker(const edm::ParameterSet& conf):
      CkfTrackCandidateMakerBase(conf, consumesCollector())
    {
      theTrackCandidateOutput=conf.getParameter<bool>("trackCandidateAlso");
      theTrajectoryOutput=true;
      if (theTrackCandidateOutput)
	produces<TrackCandidateCollection>();
      produces<TrajectoryCollection>();
      produces<std::vector<SeedStopInfo> >();
    }

    ~CkfTrajectoryMaker() override{;}

    void beginRun (edm::Run const & run, edm::EventSetup const & es) override {beginRunBase(run,es);}

    void produce(edm::Event& e, const edm::EventSetup& es) override {produceBase(e,es);}
  };
}

#endif

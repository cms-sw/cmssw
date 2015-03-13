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

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"


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
    }

    virtual ~CkfTrajectoryMaker(){;}

    virtual void beginRun (edm::Run const & run, edm::EventSetup const & es) override {beginRunBase(run,es);}

    virtual void produce(edm::Event& e, const edm::EventSetup& es) override {produceBase(e,es);}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      desc.add<edm::InputTag>("src",edm::InputTag("globalMixedSeeds"))->setComment("");
      desc.add<edm::InputTag>("MeasurementTrackerEvent",edm::InputTag("MeasurementTrackerEvent"))->setComment("");

      desc.add<bool>("reverseTrajectories",false)->setComment("");
      desc.add<bool>("cleanTrajectoryAfterInOut",false)->setComment("");
      desc.add<bool>("useHitsSplitting",false)->setComment("");
      desc.add<bool>("doSeedingRegionRebuilding",false)->setComment("");
      desc.add<bool>("trackCandidateAlso",false)->setComment("");
      {
	edm::ParameterSetDescription psd0;
	psd0.add<std::string>("propagatorAlongTISE","PropagatorWithMaterial")->setComment("");
	psd0.add<int>("numberMeasurementsForFit",4)->setComment("");
	psd0.add<std::string>("propagatorOppositeTISE","PropagatorWithMaterialOpposite")->setComment("");

	desc.add<edm::ParameterSetDescription>("TransientInitialStateEstimatorParameters",psd0)->setComment("");
      }
      desc.add<std::string>("TrajectoryCleaner","TrajectoryCleanerBySharedHits")->setComment("");
      desc.add<std::string>("RedundantSeedCleaner","CachingSeedCleanerBySharedInput")->setComment("");
      desc.add<std::string>("TrajectoryBuilder","GroupedCkfTrajectoryBuilder")->setComment("");
      {
	edm::ParameterSetDescription psd0;
	psd0.add<std::string>("refToPSet_","GroupedCkfTrajectoryBuilder")->setComment("");

	desc.add<edm::ParameterSetDescription>("TrajectoryBuilderPSet",psd0)->setComment("");
      }
      desc.add<std::string>("NavigationSchool","SimpleNavigationSchool")->setComment("");
      desc.add<unsigned int>("maxNSeeds",100000)->setComment("");

      descriptions.add("ckfTrajectories",desc);
      descriptions.setComment("");
    }


  };
}

#endif

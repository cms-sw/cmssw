#ifndef CkfTrackCandidateMaker_h
#define CkfTrackCandidateMaker_h

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

class TransientInitialStateEstimator;

namespace cms
{
  class dso_internal CkfTrackCandidateMaker : public edm::stream::EDProducer<>, public CkfTrackCandidateMakerBase
  {
  public:

    explicit CkfTrackCandidateMaker(const edm::ParameterSet& conf):
      CkfTrackCandidateMakerBase(conf, consumesCollector()){
      produceSeedStopReasons_ = conf.getParameter<bool>("produceSeedStopReasons");
      produces<TrackCandidateCollection>();
      if(produceSeedStopReasons_) {
        produces<std::vector<short> >();
      }
    }

    virtual ~CkfTrackCandidateMaker(){;}

    virtual void beginRun (edm::Run const& r, edm::EventSetup const & es) override {beginRunBase(r,es);}

    virtual void produce(edm::Event& e, const edm::EventSetup& es) override {produceBase(e,es);}
    
  };
}

#endif


#ifndef CkfDebugTrackCandidateMaker_h
#define CkfDebugTrackCandidateMaker_h

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "RecoTracker/DebugTools/interface/CkfDebugTrajectoryBuilder.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace cms {
  class CkfDebugTrackCandidateMaker : public edm::EDProducer, public CkfTrackCandidateMakerBase {
  public:
    CkfDebugTrackCandidateMaker(const edm::ParameterSet& conf) : CkfTrackCandidateMakerBase(conf, consumesCollector()) {
      produces<TrackCandidateCollection>();
    }

    virtual void beginRun (edm::Run const & run, edm::EventSetup const & es) override {
      beginRunBase(run,es); 
      initDebugger(es);
    }

    virtual void produce(edm::Event& e, const edm::EventSetup& es) override {produceBase(e,es);}
    virtual void endJob() override {delete dbg; }

  private:
    virtual TrajectorySeedCollection::const_iterator 
      lastSeed(TrajectorySeedCollection& theSeedColl){return theSeedColl.begin()+1;}

    void initDebugger(edm::EventSetup const & es){
      dbg = new CkfDebugger(es, consumesCollector());
      myTrajectoryBuilder = dynamic_cast<const CkfDebugTrajectoryBuilder*>(theTrajectoryBuilder.get());
      if (myTrajectoryBuilder) myTrajectoryBuilder->setDebugger( dbg);
      else throw cms::Exception("CkfDebugger") << "please use CkfDebugTrajectoryBuilder";
	//theTrajectoryBuilder->setDebugger( dbg);
    };
    
    void printHitsDebugger(edm::Event& e) override{dbg->printSimHits(e);};
    void countSeedsDebugger() override{dbg->countSeed();};
    void deleteAssocDebugger() override{dbg->deleteHitAssociator();};
    void deleteDebugger(){delete dbg;};
    CkfDebugger *  dbg;
    const CkfDebugTrajectoryBuilder* myTrajectoryBuilder;
  };
}

#endif

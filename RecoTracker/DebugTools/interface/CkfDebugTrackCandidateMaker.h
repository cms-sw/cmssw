#ifndef CkfDebugTrackCandidateMaker_h
#define CkfDebugTrackCandidateMaker_h

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"
#include "RecoTracker/DebugTools/interface/CkfDebugTrajectoryBuilder.h"

namespace cms {
  class CkfDebugTrackCandidateMaker : public CkfTrackCandidateMaker {
  public:
    CkfDebugTrackCandidateMaker(const edm::ParameterSet& conf) : CkfTrackCandidateMaker(conf) {;}
  private:
    virtual TrajectorySeedCollection::const_iterator lastSeed(TrajectorySeedCollection& theSeedColl){return theSeedColl.begin()+1;}
    void initDebugger(edm::EventSetup const & es){
      dbg = new CkfDebugger(es);
      ((CkfDebugTrajectoryBuilder*) getTrajectoryBuilder())->setDebugger( dbg);
    };
    void printHitsDebugger(edm::Event& e){dbg->printSimHits(e);};
    void countSeedsDebugger(){dbg->countSeed();};
    void deleteAssocDebugger(){dbg->deleteHitAssociator();};
    void deleteDebugger(){delete dbg;};
    CkfDebugger *  dbg;
  };
}

#endif

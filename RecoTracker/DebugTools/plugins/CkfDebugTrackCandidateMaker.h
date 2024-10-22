
#ifndef CkfDebugTrackCandidateMaker_h
#define CkfDebugTrackCandidateMaker_h

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "CkfDebugTrajectoryBuilder.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include <memory>

//CkfDebugger wants to see all the events and can only handle a single thread at a time
namespace cms {
  class CkfDebugTrackCandidateMaker : public edm::one::EDProducer<edm::one::WatchRuns>,
                                      public CkfTrackCandidateMakerBase {
  public:
    CkfDebugTrackCandidateMaker(const edm::ParameterSet& conf) : CkfTrackCandidateMakerBase(conf, consumesCollector()) {
      produces<TrackCandidateCollection>();
      produces<SeedStopInfo>();
      dbg = std::make_unique<CkfDebugger>(consumesCollector());
    }

    void beginRun(edm::Run const& run, edm::EventSetup const& es) override {
      beginRunBase(run, es);
      initDebugger(es);
    }
    void endRun(edm::Run const&, edm::EventSetup const&) override {}

    void produce(edm::Event& e, const edm::EventSetup& es) override { produceBase(e, es); }

  private:
    TrajectorySeedCollection::const_iterator lastSeed(TrajectorySeedCollection const& theSeedColl) override {
      return theSeedColl.begin() + 1;
    }

    void initDebugger(edm::EventSetup const& es) {
      dbg->setConditions(es);
      myTrajectoryBuilder = dynamic_cast<const CkfDebugTrajectoryBuilder*>(theTrajectoryBuilder.get());
      if (myTrajectoryBuilder)
        myTrajectoryBuilder->setDebugger(dbg.get());
      else
        throw cms::Exception("CkfDebugger") << "please use CkfDebugTrajectoryBuilder";
      //theTrajectoryBuilder->setDebugger( dbg);
    };

    void printHitsDebugger(edm::Event& e) override { dbg->printSimHits(e); };
    void countSeedsDebugger() override { dbg->countSeed(); };
    void deleteAssocDebugger() override { dbg->deleteHitAssociator(); };
    void deleteDebugger() { dbg.reset(); };
    std::unique_ptr<CkfDebugger> dbg;
    const CkfDebugTrajectoryBuilder* myTrajectoryBuilder;
  };
}  // namespace cms

#endif

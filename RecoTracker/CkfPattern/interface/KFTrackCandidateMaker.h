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

namespace cms
{
  class KFTrackCandidateMaker : public edm::EDProducer
  {
  public:

    explicit KFTrackCandidateMaker(const edm::ParameterSet& conf);

    virtual ~KFTrackCandidateMaker();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CombinatorialTrajectoryBuilder   theCombinatorialTrajectoryBuilder;
    TrajectoryCleaner*               theTrajectoryCleaner;

    edm::ParameterSet conf_;
    int isInitialized;

  };
}

#endif

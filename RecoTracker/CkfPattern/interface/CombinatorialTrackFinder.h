#ifndef CombinatorialTrackFinder_h
#define CombinatorialTrackFinder_h



#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/CkfPattern/interface/CombinatorialTrajectoryBuilder.h"



namespace cms
{
  class CombinatorialTrackFinder : public edm::EDProducer
  {
  public:

    explicit CombinatorialTrackFinder(const edm::ParameterSet& conf);

    virtual ~CombinatorialTrackFinder();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    CombinatorialTrajectoryBuilder combinatorialTrajectoryBuilder_;
    edm::ParameterSet conf_;
    int isInitialized;

  };
}

#endif

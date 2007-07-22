#ifndef RecoPixelVertexing_PixelLowPtUtilities_SiPixelRecHitRemover_h
#define RecoPixelVertexing_PixelLowPtUtilities_SiPixelRecHitRemover_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include <vector>
using namespace std;

class SiPixelRecHitRemover : public edm::EDProducer
{
  public:
    explicit SiPixelRecHitRemover(const edm::ParameterSet& ps);
    ~SiPixelRecHitRemover();

//    virtual void beginJob(const edm::EventSetup& es);
    virtual void produce(edm::Event& ev, const edm::EventSetup& es);

  private:
//    edm::InputTag hitCollectionLabel;
    std::string hitCollectionLabel;
    std::vector<std::string> removeHitsList;
};

#endif

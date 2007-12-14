#ifndef _RecHitRemover_h_
#define _RecHitRemover_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include <vector>
using namespace std;

class RecHitRemover
{
  public:
    RecHitRemover(const edm::ParameterSet& ps);
    ~RecHitRemover();
    SiPixelRecHitCollection getFreeHits(const edm::Event& ev);

  private:
    edm::InputTag hitCollectionLabel;
    std::vector<std::string> removeHitsList;
};

#endif

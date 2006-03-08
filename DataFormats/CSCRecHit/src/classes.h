#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <FWCore/EDProduct/interface/Wrapper.h>
#include <vector>

namespace{ 
  namespace {
    CSCRecHit2DCollection dc;
    edm::Wrapper<CSCRecHit2DCollection> dwc;
  }
}

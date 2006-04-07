#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>

namespace{ 
  namespace {
    CSCRecHit2DCollection dc;
    edm::Wrapper<CSCRecHit2DCollection> dwc;
  }
}

namespace{ 
  namespace {
    std::vector<CSCSegment> seg;    
    edm::Wrapper<std::vector<CSCSegment> > dwc1;
  }
}

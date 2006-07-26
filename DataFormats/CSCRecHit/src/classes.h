#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCStripHit.h>
#include <DataFormats/CSCRecHit/interface/CSCStripHitCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCWireHit.h>
#include <DataFormats/CSCRecHit/interface/CSCWireHitCollection.h>
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
    CSCStripHitCollection dc0;
    edm::Wrapper<CSCStripHitCollection> dwc0;
  }
}

namespace{
  namespace {
    CSCStripHitCollection dc2;
    edm::Wrapper<CSCWireHitCollection> dwc2;
  }
}

namespace{ 
  namespace {
    CSCSegmentCollection seg;    
    edm::Wrapper<CSCSegmentCollection> dwc1;
  }
}

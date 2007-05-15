#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>

namespace{ 
  namespace {
    std::map<CSCDetId,std::pair<unsigned int,unsigned int> > dummycscdetid1;  
    std::map<CSCDetId,std::pair<unsigned long,unsigned long> > dummycscdetid2;  
  }
}

namespace{ 
  namespace {
    CSCRecHit2DCollection dc;
    edm::Wrapper<CSCRecHit2DCollection> dwc;
  }
}

namespace{ 
  namespace {
    CSCSegmentCollection seg;    
    edm::Wrapper<CSCSegmentCollection> dwc1;
  }
}

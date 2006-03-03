#include "DataFormats/CSCRec2Hit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace{ 
  namespace {
    CSCRecHit2D dh;

    std::vector<CSCRecHit2D>  dv;
    
    edm::ClonePolicy<CSCRecHit2D> dcp;
    edm::OwnVector<CSCRecHit2D,edm::ClonePolicy<CSCRecHit2D> > dov;

    CSCRecHit2DCollection dc;
    edm::Wrapper<CSCRecHitCollection> dwc;
  }

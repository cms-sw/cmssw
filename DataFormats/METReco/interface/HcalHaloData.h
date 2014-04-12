#ifndef DATAFORMATS_METRECO_HCALHALODATA_H
#define DATAFORMATS_METRECO_HCALHALODATA_H
/*
  [class]:  HcalHaloData
  [authors]: R. Remington, The University of Florida
  [description]: Container class to store beam halo data specific to the Hcal subdetector
  [date]: October 15, 2009
*/
#include <vector>
#include "DataFormats/METReco/interface/PhiWedge.h"

namespace reco {
  class HcalHaloData {
  public:
    //constructor
    HcalHaloData();
    //destructor
    ~HcalHaloData(){}
    
    // Return collection of 5-degree Phi Wedges built from Hcal RecHits
    const std::vector<PhiWedge>& GetPhiWedges() const {return PhiWedgeCollection;}
    std::vector<PhiWedge>& GetPhiWedges()  {return PhiWedgeCollection;}
    
  private:
    std::vector<PhiWedge> PhiWedgeCollection;
    
  };
}
#endif

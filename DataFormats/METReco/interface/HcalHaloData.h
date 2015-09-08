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
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

struct HaloTowerStrip {
    std::vector<std::pair<uint8_t, CaloTowerDetId> > cellTowerIds;
    float hadEt;
    HaloTowerStrip() {
        hadEt = 0.0;
        cellTowerIds.clear();
    }
    HaloTowerStrip(const HaloTowerStrip& strip) {
        cellTowerIds = strip.cellTowerIds;
        hadEt = strip.hadEt;
    }
};

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

    // Return collection of problematic strips (pairs of # of problematic HCAL cells and CaloTowerDetId)
    const std::vector<HaloTowerStrip>& getProblematicStrips() const {return problematicStripCollection;}
    std::vector<HaloTowerStrip>& getProblematicStrips()  {return problematicStripCollection;}

  private:
    std::vector<PhiWedge> PhiWedgeCollection;
    std::vector<HaloTowerStrip> problematicStripCollection;
    
  };
}
#endif

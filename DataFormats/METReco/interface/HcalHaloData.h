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
    const std::vector<std::vector<std::pair<char, CaloTowerDetId> > >& GetProblematicStrips() const {return ProblematicStripCollection;}
    std::vector<std::vector<std::pair<char, CaloTowerDetId> > >& GetProblematicStrips()  {return ProblematicStripCollection;}

    // Total energies of problematic strips (HCAL)
    const std::vector<float> & GetProblematicStripsHadEt() const {return ProblematicStripHadEt;}
    std::vector<float> & GetProblematicStripsHadEt() {return ProblematicStripHadEt;}

  private:
    std::vector<PhiWedge> PhiWedgeCollection;
    std::vector<std::vector<std::pair<char, CaloTowerDetId> > > ProblematicStripCollection;
    std::vector<float> ProblematicStripHadEt;
    
  };
}
#endif

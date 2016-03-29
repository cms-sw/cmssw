#ifndef UCTRegion_hh
#define UCTRegion_hh

#include <vector>
#include <iostream>

#include "UCTTower.hh"

namespace l1tcalo {
  constexpr uint32_t RegionETMask{ 0x000003FF};
  constexpr uint32_t RegionLocBits{0x0000F000};
  constexpr uint32_t LocationShift{12};
}

class UCTRegion {
public:

  UCTRegion(uint32_t crt, uint32_t crd, bool ne, uint32_t rgn);

  virtual ~UCTRegion();

  // To setData for towers before processing

  const std::vector<UCTTower*>& getTowers() {return towers;}

  // To process event

  bool clearEvent();
  bool setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET);
  bool setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET);
  bool process();

  // Packed data access

  const uint32_t rawData() const {return regionSummary;}
  const uint32_t location() const {return ((regionSummary & l1tcalo::RegionLocBits) >> l1tcalo::LocationShift);}

  const int hitCaloEta() const {
    uint32_t highestTowerLocation = location();
    return towers[highestTowerLocation]->caloEta();
  }

  const int hitCaloPhi() const {
    uint32_t highestTowerLocation = (location() & 0xF);
    return towers[highestTowerLocation]->caloPhi();
  }

  const UCTTowerIndex hitTowerIndex() const {
    return UCTTowerIndex(hitCaloEta(), hitCaloPhi());
  }

  const UCTRegionIndex regionIndex() const {
    UCTGeometry g;
    return UCTRegionIndex(g.getUCTRegionEtaIndex(negativeEta, region), g.getUCTRegionPhiIndex(crate, card));
  }

  const uint32_t compressedData() const {return regionSummary;}

  // Access functions for convenience
  // Note that the bit fields are limited in hardware

  const uint32_t et() const {return (l1tcalo::RegionETMask & regionSummary);}

  // More access functions

  const uint32_t getCrate() const {return crate;}
  const uint32_t getCard() const {return card;}
  const uint32_t getRegion() const {return region;}

  const bool isNegativeEta() const {return negativeEta;}

  const UCTTower* getTower(UCTTowerIndex t) const {
    return getTower(t.first, t.second);
  }

  friend std::ostream& operator<<(std::ostream&, const UCTRegion&);
  
private:

  // No default constructor is needed

  UCTRegion();

  // No copy constructor is needed

  UCTRegion(const UCTRegion&);

  // No equality operator is needed

  const UCTRegion& operator=(const UCTRegion&);

protected:

  // Helper functions

  const UCTTower* getTower(uint32_t caloEta, uint32_t caloPhi) const;

  // Region location definition

  uint32_t crate;
  uint32_t card;
  uint32_t region;
  bool negativeEta;

  // Owned region level data 

  std::vector<UCTTower*> towers;

  uint32_t regionSummary;

};

#endif

#ifndef UCTTower_hh
#define UCTTower_hh

#include "UCTGeometry.hh"

#include <vector>

namespace l1tcalo {
  constexpr uint32_t etInputMax{0xFF};

  constexpr uint32_t etMask{0x000001FF};
  constexpr uint32_t erMask{0x00000E00};
  constexpr uint32_t erMaxV{7};
  constexpr uint32_t erShift{9};
  constexpr uint32_t zeroFlagMask{0x00001000};
  constexpr uint32_t eohrFlagMask{0x00002000};
  constexpr uint32_t hcalFlagMask{0x00004000};
  constexpr uint32_t ecalFlagMask{0x00008000};
  constexpr uint32_t stg2BitsMask{0x0000FFFF};
  constexpr uint32_t miscBitsMask{0x0000F000};
  constexpr uint32_t miscShift{12};

  constexpr uint32_t ecalBitsMask{0x00FF0000};
  constexpr uint32_t ecalShift{16};
  constexpr uint32_t hcalBitsMask{0xFF000000};
  constexpr uint32_t hcalShift{24};
}

class UCTLayer1;

class UCTTower {
public:

  UCTTower(uint32_t crt, uint32_t crd, bool ne, uint32_t rgn, uint32_t eta, uint32_t phi) :
    crate(crt),
    card(crd),
    region(rgn),
    iEta(eta),
    iPhi(phi),
    negativeEta(ne),
    ecalFG(false),
    ecalET(0),
    hcalET(0),
    hcalFB(0),
    ecalLUT(0),
    hcalLUT(0),
    hfLUT(0),
    towerData(0)
  {}

  UCTTower(uint16_t location);
  
  virtual ~UCTTower() {;}

  bool clearEvent() {
    ecalFG = false;
    ecalET = 0;
    hcalET = 0;
    hcalFB = 0;
    towerData = 0;
    return true;
  }

  bool setECALData(bool ecalFG, uint32_t ecalET);
  bool setHCALData(uint32_t hcalFB, uint32_t hcalET);
  bool setHFData(uint32_t fbIn, uint32_t etIn);

  bool setECALLUT(const std::vector< std::vector< std::vector< uint32_t > > > *l) {
    ecalLUT = l;
    return true;
  }
  
  bool setHCALLUT(const std::vector< std::vector< std::vector< uint32_t > > > *l) {
    hcalLUT = l;
    return true;
  }
  
  bool setHFLUT(const std::vector< std::vector< uint32_t > > *l) {
    hfLUT = l;
    return true;
  }

  bool process();
  bool processHFTower();

  // Packed data access

  const uint32_t rawData() const {return towerData;}
  const uint16_t location() const;
  const uint64_t extendedData() const;
  const uint16_t compressedData() const {return (uint16_t) (towerData & l1tcalo::stg2BitsMask);}

  // Access functions for convenience
  // Note that the bit fields are limited in hardware

  const uint32_t et() const {return (towerData & l1tcalo::etMask);}
  const uint32_t er() const {return ((towerData & l1tcalo::erMask) >> l1tcalo::erShift);}
  const uint8_t miscBits() const {return (uint8_t) ((towerData & l1tcalo::miscBitsMask) >> l1tcalo::miscShift);}

  const uint32_t getEcalET() const {return ((towerData & l1tcalo::ecalBitsMask) >> l1tcalo::ecalShift);}
  const uint32_t getHcalET() const {return ((towerData & l1tcalo::hcalBitsMask) >> l1tcalo::hcalShift);}

  const bool zeroFlag() const {return ((towerData & l1tcalo::zeroFlagMask) == l1tcalo::zeroFlagMask);}
  const bool eohrFlag() const {return ((towerData & l1tcalo::eohrFlagMask) == l1tcalo::eohrFlagMask);}
  const bool hcalFlag() const {return ((towerData & l1tcalo::hcalFlagMask) == l1tcalo::hcalFlagMask);}
  const bool ecalFlag() const {return ((towerData & l1tcalo::ecalFlagMask) == l1tcalo::ecalFlagMask);}

  // More access functions

  const uint32_t getCrate() const {return crate;}
  const uint32_t getCard() const {return card;}
  const uint32_t getRegion() const {return region;}
  const uint32_t getiEta() const {return iEta;}
  const uint32_t getiPhi() const {return iPhi;}
  const bool isNegativeEta() const {return negativeEta;}

  const int caloEta() const {
    UCTGeometry g;
    return g.getCaloEtaIndex(negativeEta, region, iEta);
  }

  const int caloPhi() const {
    UCTGeometry g;
    return g.getCaloPhiIndex(crate, card, region, iPhi);
  }

  const UCTTowerIndex towerIndex() const {
    return UCTTowerIndex(caloEta(), caloPhi());
  }

  friend std::ostream& operator<<(std::ostream&, const UCTTower&);

private:

  // No default constructor is needed

  UCTTower();

  // No copy constructor is needed

  UCTTower(const UCTTower&);

  // No equality operator is needed

  const UCTTower& operator=(const UCTTower&);

  // Tower location definition

  uint32_t crate;
  uint32_t card;
  uint32_t region;
  uint32_t iEta;
  uint32_t iPhi;
  bool negativeEta;

  // Input data

  bool ecalFG;
  uint32_t ecalET;
  uint32_t hcalET;
  uint32_t hcalFB;

  // Lookup table

  const std::vector< std::vector< std::vector< uint32_t > > > *ecalLUT;
  const std::vector< std::vector< std::vector< uint32_t > > > *hcalLUT;
  const std::vector< std::vector< uint32_t > > *hfLUT;
  
  // Owned tower level data 
  // Packed bits -- only bottom 16 bits are used in "prelim" protocol

  uint32_t towerData;

};

#endif

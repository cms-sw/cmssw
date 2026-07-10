#ifndef DataFormats_L1Scouting_L1ScoutingCaloTower_h
#define DataFormats_L1Scouting_L1ScoutingCaloTower_h

#include <cstdint>
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class CaloTower {
  public:
    CaloTower() : hwEt_(0), erBits_(0), miscBits_(0), hwEta_(0), hwPhi_(0) {}

    CaloTower(int16_t hwEt, int16_t erBits, int16_t miscBits, int16_t hwEta, int16_t hwPhi)
        : hwEt_(hwEt), erBits_(erBits), miscBits_(miscBits), hwEta_(hwEta), hwPhi_(hwPhi) {}

    void setHwEt(int16_t hwEt) { hwEt_ = hwEt; }
    void setErBits(int16_t erBits) { erBits_ = erBits; }
    void setMiscBits(int16_t miscBits) { miscBits_ = miscBits; }
    void setHwEta(int16_t hwEta) { hwEta_ = hwEta; }
    void setHwPhi(int16_t hwPhi) { hwPhi_ = hwPhi; }

    int16_t hwEt() const { return hwEt_; }
    int16_t erBits() const { return erBits_; }
    int16_t miscBits() const { return miscBits_; }
    int16_t hwEta() const { return hwEta_; }
    int16_t hwPhi() const { return hwPhi_; }

  private:
    int16_t hwEt_;
    int16_t erBits_;
    int16_t miscBits_;
    int16_t hwEta_;
    int16_t hwPhi_;
  };

  typedef OrbitCollection<CaloTower> CaloTowerOrbitCollection;

}  // namespace l1ScoutingRun3
#endif  // DataFormats_L1Scouting_L1ScoutingCaloTower_h

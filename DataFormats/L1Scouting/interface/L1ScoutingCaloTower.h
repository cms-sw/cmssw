#ifndef DataFormats_L1Scouting_L1ScoutingCaloTower_h
#define DataFormats_L1Scouting_L1ScoutingCaloTower_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class CaloTower {
  public:
    CaloTower() : hwEt_(0), erBits_(0), miscBits_(0), hwEta_(0), hwPhi_(0) {}

    CaloTower(int hwEt, int erBits, int miscBits, int hwPhi, int hwEta) : hwEt_(hwEt), erBits_(erBits), miscBits_(miscBits), hwEta_(hwEta), hwPhi_(hwPhi) {}

    void setHwEt(int hwEt) { hwEt_ = hwEt; }
    void setErBits(int erBits) { erBits_ = erBits; }
    void setMiscBits(int miscBits) { miscBits_ = miscBits; }
    void setHwEta(int hwEta) { hwEta_ = hwEta; }
    void setHwPhi(int hwPhi) { hwPhi_ = hwPhi; }

    int hwEt() const { return hwEt_; }
    int erBits() const { return erBits_; }
    int miscBits() const { return miscBits_; }
    int hwEta() const { return hwEta_; }
    int hwPhi() const { return hwPhi_; }

  private:
    int hwEt_;
    int erBits_;
    int miscBits_;
    int hwEta_;
    int hwPhi_;
  };

  typedef OrbitCollection<CaloTower> CaloTowerOrbitCollection;

}  // namespace l1ScoutingRun3
#endif  // DataFormats_L1Scouting_L1ScoutingCaloTower_h

#ifndef DataFormats_L1Scouting_L1ScoutingCalo_h
#define DataFormats_L1Scouting_L1ScoutingCalo_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class CaloObject {
  public:
    CaloObject() : hwEt_(0), hwEta_(0), hwPhi_(0), hwIso_(0) {}

    CaloObject(int hwEt, int hwEta, int hwPhi, int iso) : hwEt_(hwEt), hwEta_(hwEta), hwPhi_(hwPhi), hwIso_(iso) {}

    void setHwEt(int hwEt) { hwEt_ = hwEt; }
    void setHwEta(int hwEta) { hwEta_ = hwEta; }
    void setHwPhi(int hwPhi) { hwPhi_ = hwPhi; }
    void setHwIso(int hwIso) { hwIso_ = hwIso; }

    int hwEt() const { return hwEt_; }
    int hwEta() const { return hwEta_; }
    int hwPhi() const { return hwPhi_; }
    int hwIso() const { return hwIso_; }

  private:
    int hwEt_;
    int hwEta_;
    int hwPhi_;
    int hwIso_;
  };

  class Jet : public CaloObject {
  public:
    Jet() : CaloObject(0, 0, 0, 0) {}

    Jet(int hwEt, int hwEta, int hwPhi, int hwQual) : CaloObject(hwEt, hwEta, hwPhi, hwQual) {}

    // store quality instead of iso
    void setHwQual(int hwQual) { setHwIso(hwQual); }
    int hwQual() const { return hwIso(); }
  };

  class EGamma : public CaloObject {
  public:
    EGamma() : CaloObject(0, 0, 0, 0) {}

    EGamma(int hwEt, int hwEta, int hwPhi, int iso) : CaloObject(hwEt, hwEta, hwPhi, iso) {}
  };

  class Tau : public CaloObject {
  public:
    Tau() : CaloObject(0, 0, 0, 0) {}

    Tau(int hwEt, int hwEta, int hwPhi, int iso) : CaloObject(hwEt, hwEta, hwPhi, iso) {}
  };

  class BxSums {
  public:
    BxSums()
        : hwTotalEt_(0),
          hwTotalEtEm_(0),
          hwTotalHt_(0),
          hwMissEt_(0),
          hwMissEtPhi_(0),
          hwMissHt_(0),
          hwMissHtPhi_(0),
          hwMissEtHF_(0),
          hwMissEtHFPhi_(0),
          hwMissHtHF_(0),
          hwMissHtHFPhi_(0),
          hwAsymEt_(0),
          hwAsymHt_(0),
          hwAsymEtHF_(0),
          hwAsymHtHF_(0),
          minBiasHFP0_(0),
          minBiasHFM0_(0),
          minBiasHFP1_(0),
          minBiasHFM1_(0),
          towerCount_(0),
          centrality_(0) {}

    BxSums(int hwTotalEt,
           int hwTotalEtEm,
           int hwTotalHt,
           int hwMissEt,
           int hwMissEtPhi,
           int hwMissHt,
           int hwMissHtPhi,
           int hwMissEtHF,
           int hwMissEtHFPhi,
           int hwMissHtHF,
           int hwMissHtHFPhi,
           int hwAsymEt,
           int hwAsymHt,
           int hwAsymEtHF,
           int hwAsymHtHF,
           int minBiasHFP0,
           int minBiasHFM0,
           int minBiasHFP1,
           int minBiasHFM1,
           int towerCount,
           int centrality)
        : hwTotalEt_(hwTotalEt),
          hwTotalEtEm_(hwTotalEtEm),
          hwTotalHt_(hwTotalHt),
          hwMissEt_(hwMissEt),
          hwMissEtPhi_(hwMissEtPhi),
          hwMissHt_(hwMissHt),
          hwMissHtPhi_(hwMissHtPhi),
          hwMissEtHF_(hwMissEtHF),
          hwMissEtHFPhi_(hwMissEtHFPhi),
          hwMissHtHF_(hwMissHtHF),
          hwMissHtHFPhi_(hwMissHtHFPhi),
          hwAsymEt_(hwAsymEt),
          hwAsymHt_(hwAsymHt),
          hwAsymEtHF_(hwAsymEtHF),
          hwAsymHtHF_(hwAsymHtHF),
          minBiasHFP0_(minBiasHFP0),
          minBiasHFM0_(minBiasHFM0),
          minBiasHFP1_(minBiasHFP1),
          minBiasHFM1_(minBiasHFM1),
          towerCount_(towerCount),
          centrality_(centrality) {}

    void setHwTotalEt(int hwTotalEt) { hwTotalEt_ = hwTotalEt; }
    void setHwTotalEtEm(int hwTotalEtEm) { hwTotalEtEm_ = hwTotalEtEm; }
    void setMinBiasHFP0(int minBiasHFP0) { minBiasHFP0_ = minBiasHFP0; }
    void setHwTotalHt(int hwTotalHt) { hwTotalHt_ = hwTotalHt; }
    void setTowerCount(int towerCount) { towerCount_ = towerCount; }
    void setMinBiasHFM0(int minBiasHFM0) { minBiasHFM0_ = minBiasHFM0; }
    void setHwMissEt(int hwMissEt) { hwMissEt_ = hwMissEt; }
    void setHwMissEtPhi(int hwMissEtPhi) { hwMissEtPhi_ = hwMissEtPhi; }
    void setHwAsymEt(int hwAsymEt) { hwAsymEt_ = hwAsymEt; }
    void setMinBiasHFP1(int minBiasHFP1) { minBiasHFP1_ = minBiasHFP1; }
    void setHwMissHt(int hwMissHt) { hwMissHt_ = hwMissHt; }
    void setHwMissHtPhi(int hwMissHtPhi) { hwMissHtPhi_ = hwMissHtPhi; }
    void setHwAsymHt(int hwAsymHt) { hwAsymHt_ = hwAsymHt; }
    void setMinBiasHFM1(int minBiasHFM1) { minBiasHFM1_ = minBiasHFM1; }
    void setHwMissEtHF(int hwMissEtHF) { hwMissEtHF_ = hwMissEtHF; }
    void setHwMissEtHFPhi(int hwMissEtHFPhi) { hwMissEtHFPhi_ = hwMissEtHFPhi; }
    void setHwAsymEtHF(int hwAsymEtHF) { hwAsymEtHF_ = hwAsymEtHF; }
    void setHwMissHtHF(int hwMissHtHF) { hwMissHtHF_ = hwMissHtHF; }
    void setHwMissHtHFPhi(int hwMissHtHFPhi) { hwMissHtHFPhi_ = hwMissHtHFPhi; }
    void setHwAsymHtHF(int hwAsymHtHF) { hwAsymHtHF_ = hwAsymHtHF; }
    void setCentrality(int centrality) { centrality_ = centrality; }

    const int hwTotalEt() const { return hwTotalEt_; }
    const int hwTotalEtEm() const { return hwTotalEtEm_; }
    const int minBiasHFP0() const { return minBiasHFP0_; }
    const int hwTotalHt() const { return hwTotalHt_; }
    const int towerCount() const { return towerCount_; }
    const int minBiasHFM0() const { return minBiasHFM0_; }
    const int hwMissEt() const { return hwMissEt_; }
    const int hwMissEtPhi() const { return hwMissEtPhi_; }
    const int hwAsymEt() const { return hwAsymEt_; }
    const int minBiasHFP1() const { return minBiasHFP1_; }
    const int hwMissHt() const { return hwMissHt_; }
    const int hwMissHtPhi() const { return hwMissHtPhi_; }
    const int hwAsymHt() const { return hwAsymHt_; }
    const int minBiasHFM1() const { return minBiasHFM1_; }
    const int hwMissEtHF() const { return hwMissEtHF_; }
    const int hwMissEtHFPhi() const { return hwMissEtHFPhi_; }
    const int hwAsymEtHF() const { return hwAsymEtHF_; }
    const int hwMissHtHF() const { return hwMissHtHF_; }
    const int hwMissHtHFPhi() const { return hwMissHtHFPhi_; }
    const int hwAsymHtHF() const { return hwAsymHtHF_; }
    const int centrality() const { return centrality_; }

  private:
    int hwTotalEt_;
    int hwTotalEtEm_;
    int hwTotalHt_;
    int hwMissEt_;
    int hwMissEtPhi_;
    int hwMissHt_;
    int hwMissHtPhi_;
    int hwMissEtHF_;
    int hwMissEtHFPhi_;
    int hwMissHtHF_;
    int hwMissHtHFPhi_;
    int hwAsymEt_;
    int hwAsymHt_;
    int hwAsymEtHF_;
    int hwAsymHtHF_;
    int minBiasHFP0_;
    int minBiasHFM0_;
    int minBiasHFP1_;
    int minBiasHFM1_;
    int towerCount_;
    int centrality_;
  };

  typedef OrbitCollection<Jet> JetOrbitCollection;
  typedef OrbitCollection<EGamma> EGammaOrbitCollection;
  typedef OrbitCollection<Tau> TauOrbitCollection;
  typedef OrbitCollection<BxSums> BxSumsOrbitCollection;

}  // namespace l1ScoutingRun3
#endif  // DataFormats_L1Scouting_L1ScoutingCalo_h
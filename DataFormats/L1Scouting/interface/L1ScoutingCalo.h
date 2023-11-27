#ifndef DataFormats_L1Scouting_L1ScoutingCalo_h
#define DataFormats_L1Scouting_L1ScoutingCalo_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

namespace l1ScoutingRun3 {

  class ScJet;
  typedef OrbitCollection<ScJet>    ScJetOrbitCollection;
  class ScEGamma;
  typedef OrbitCollection<ScEGamma> ScEGammaOrbitCollection;
  class ScTau;
  typedef OrbitCollection<ScTau>    ScTauOrbitCollection;
  class ScEtSum;
  typedef OrbitCollection<ScEtSum>  ScEtSumOrbitCollection;
  class ScBxSums;
  typedef OrbitCollection<ScBxSums> ScBxSumsOrbitCollection;

  class ScCaloObject {
  public:
    ScCaloObject()
    : hwEt_(0),
      hwEta_(0),
      hwPhi_(0),
      hwIso_(0){}

    ScCaloObject(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : hwEt_(hwEt),
      hwEta_(hwEta),
      hwPhi_(hwPhi),
      hwIso_(iso) {}

    ScCaloObject(const ScCaloObject& other) = default;
    ScCaloObject(ScCaloObject&& other) = default;
    ScCaloObject & operator=(const ScCaloObject& other) = default;
    ScCaloObject & operator=(ScCaloObject&& other) = default;

    void swap(ScCaloObject& other){
      using std::swap;
      swap(hwEt_, other.hwEt_);
      swap(hwEta_, other.hwEta_);
      swap(hwPhi_, other.hwPhi_);
      swap(hwIso_, other.hwIso_);
    }

    inline void setHwEt(int hwEt) { hwEt_= hwEt;}
    inline void setHwEta(int hwEta) { hwEta_= hwEta;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setHwIso(int hwIso) { hwIso_= hwIso;}

    inline int hwEt() const {return hwEt_;}
    inline int hwEta() const {return hwEta_;}
    inline int hwPhi() const {return hwPhi_;}
    inline int hwIso() const {return hwIso_;}

  private:
    int hwEt_;
    int hwEta_;
    int hwPhi_;
    int hwIso_;

  };

  class ScJet: public ScCaloObject {
  public:
    ScJet(): ScCaloObject(0, 0 ,0 , 0){}

    ScJet(
      int hwEt,
      int hwEta,
      int hwPhi,
      int hwQual)
    : ScCaloObject(hwEt, hwEta ,hwPhi , hwQual) {}

    // store quality instead of iso
    inline void setHwQual(int hwQual) { setHwIso(hwQual);}
    inline int hwQual() const {return hwIso();}

  };

  class ScEGamma: public ScCaloObject {
  public:
    ScEGamma(): ScCaloObject(0, 0 ,0 , 0){}

    ScEGamma(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : ScCaloObject(hwEt, hwEta ,hwPhi , iso) {}
  };

  class ScTau: public ScCaloObject {
  public:
    ScTau(): ScCaloObject(0, 0 ,0 , 0){}

    ScTau(
      int hwEt,
      int hwEta,
      int hwPhi,
      int iso)
    : ScCaloObject(hwEt, hwEta ,hwPhi , iso) {}
  };


  class ScEtSum {
  public:
    ScEtSum()
    : hwEt_(0),
      hwPhi_(0),
      type_(l1t::EtSum::kUninitialized) {}

    ScEtSum(
      int hwEt,
      int hwPhi,
      l1t::EtSum::EtSumType type)
    : hwEt_(hwEt),
      hwPhi_(hwPhi),
      type_(type) {}

    ScEtSum(const ScEtSum& other) = default;
    ScEtSum(ScEtSum&& other) = default;
    ScEtSum & operator=(const ScEtSum& other) = default;
    ScEtSum & operator=(ScEtSum&& other) = default;

    void swap(ScEtSum& other){
      using std::swap;
      swap(hwEt_, other.hwEt_);
      swap(hwPhi_, other.hwPhi_);
      swap(type_, other.type_);
    }

    inline void setHwEt(int hwEt) { hwEt_= hwEt;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setType(l1t::EtSum::EtSumType type) { type_= type;}

    inline int hwEt() const {return hwEt_;}
    inline int hwPhi() const {return hwPhi_;}
    inline l1t::EtSum::EtSumType type() const {return type_;}

  private:
    int hwEt_;
    int hwPhi_;
    l1t::EtSum::EtSumType type_;
  };

  class ScBxSums {

  public:
    ScBxSums()
    : hwTotalEt_(0), hwTotalEtEm_(0), minBiasHFP0_(0),
      hwTotalHt_(0), towerCount_(0), minBiasHFM0_(0),
      hwMissEt_(0), hwMissEtPhi_(0), hwAsymEt_(0), minBiasHFP1_(0),
      hwMissHt_(0), hwMissHtPhi_(0), hwAsymHt_(0), minBiasHFM1_(0),
      hwMissEtHF_(0), hwMissEtHFPhi_(0), hwAsymEtHF_(0),
      hwMissHtHF_(0), hwMissHtHFPhi_(0), hwAsymHtHF_(0), centrality_(0)
    {}

    ScBxSums(
      int hwTotalEt, int hwTotalEtEm, int minBiasHFP0,
      int hwTotalHt, int towerCount, int minBiasHFM0,
      int hwMissEt, int hwMissEtPhi, int hwAsymEt, int minBiasHFP1,
      int hwMissHt, int hwMissHtPhi, int hwAsymHt, int minBiasHFM1,
      int hwMissEtHF, int hwMissEtHFPhi, int hwAsymEtHF,
      int hwMissHtHF, int hwMissHtHFPhi, int hwAsymHtHF, int centrality
    )
    : hwTotalEt_(hwTotalEt), hwTotalEtEm_(hwTotalEtEm), minBiasHFP0_(minBiasHFP0),
      hwTotalHt_(hwTotalHt), towerCount_(towerCount), minBiasHFM0_(minBiasHFM0),
      hwMissEt_(hwMissEt), hwMissEtPhi_(hwMissEtPhi), hwAsymEt_(hwAsymEt),
      minBiasHFP1_(minBiasHFP1), hwMissHt_(hwMissHt), hwMissHtPhi_(hwMissHtPhi),
      hwAsymHt_(hwAsymHt), minBiasHFM1_(minBiasHFM1), hwMissEtHF_(hwMissEtHF),
      hwMissEtHFPhi_(hwMissEtHFPhi), hwAsymEtHF_(hwAsymEtHF), hwMissHtHF_(hwMissHtHF),
      hwMissHtHFPhi_(hwMissHtHFPhi), hwAsymHtHF_(hwAsymHtHF), centrality_(centrality)
    {}

    ScBxSums(const ScBxSums& other) = default;
    ScBxSums(ScBxSums&& other) = default;
    ScBxSums & operator=(const ScBxSums& other) = default;
    ScBxSums & operator=(ScBxSums&& other) = default;

    void swap(ScBxSums& other){
      using std::swap;
      swap(hwTotalEt_, other.hwTotalEt_);
      swap(hwTotalEtEm_, other.hwTotalEtEm_);
      swap(minBiasHFP0_, other.minBiasHFP0_);
      swap(hwTotalHt_, other.hwTotalHt_);
      swap(towerCount_, other.towerCount_);
      swap(minBiasHFM0_, other.minBiasHFM0_);
      swap(hwMissEt_, other.hwMissEt_);
      swap(hwMissEtPhi_, other.hwMissEtPhi_);
      swap(hwAsymEt_, other.hwAsymEt_);
      swap(minBiasHFP1_, other.minBiasHFP1_);
      swap(hwMissHt_, other.hwMissHt_);
      swap(hwMissHtPhi_, other.hwMissHtPhi_);
      swap(hwAsymHt_, other.hwAsymHt_);
      swap(minBiasHFM1_, other.minBiasHFM1_);
      swap(hwMissEtHF_, other.hwMissEtHF_);
      swap(hwMissEtHFPhi_, other.hwMissEtHFPhi_);
      swap(hwAsymEtHF_, other.hwAsymEtHF_);
      swap(hwMissHtHF_, other.hwMissHtHF_);
      swap(hwMissHtHFPhi_, other.hwMissHtHFPhi_);
      swap(hwAsymHtHF_, other.hwAsymHtHF_);
      swap(centrality_, other.centrality_);
      
    }

    inline void setHwTotalEt(int hwTotalEt) {hwTotalEt_ = hwTotalEt;}
    inline void setHwTotalEtEm(int hwTotalEtEm) {hwTotalEtEm_ = hwTotalEtEm;}
    inline void setMinBiasHFP0(int minBiasHFP0) {minBiasHFP0_ = minBiasHFP0;}
    inline void setHwTotalHt(int hwTotalHt) {hwTotalHt_ = hwTotalHt;}
    inline void setTowerCount(int towerCount) {towerCount_ = towerCount;}
    inline void setMinBiasHFM0(int minBiasHFM0) {minBiasHFM0_ = minBiasHFM0;}
    inline void setHwMissEt(int hwMissEt) {hwMissEt_ = hwMissEt;}
    inline void setHwMissEtPhi(int hwMissEtPhi) {hwMissEtPhi_ = hwMissEtPhi;}
    inline void setHwAsymEt(int hwAsymEt) {hwAsymEt_ = hwAsymEt;}
    inline void setMinBiasHFP1(int minBiasHFP1) {minBiasHFP1_ = minBiasHFP1;}
    inline void setHwMissHt(int hwMissHt) {hwMissHt_ = hwMissHt;}
    inline void setHwMissHtPhi(int hwMissHtPhi) {hwMissHtPhi_ = hwMissHtPhi;}
    inline void setHwAsymHt(int hwAsymHt) {hwAsymHt_ = hwAsymHt;}
    inline void setMinBiasHFM1(int minBiasHFM1) {minBiasHFM1_ = minBiasHFM1;}
    inline void setHwMissEtHF(int hwMissEtHF) {hwMissEtHF_ = hwMissEtHF;}
    inline void setHwMissEtHFPhi(int hwMissEtHFPhi) {hwMissEtHFPhi_ = hwMissEtHFPhi;}
    inline void setHwAsymEtHF(int hwAsymEtHF) {hwAsymEtHF_ = hwAsymEtHF;}
    inline void setHwMissHtHF(int hwMissHtHF) {hwMissHtHF_ = hwMissHtHF;}
    inline void setHwMissHtHFPhi(int hwMissHtHFPhi) {hwMissHtHFPhi_ = hwMissHtHFPhi;}
    inline void setHwAsymHtHF(int hwAsymHtHF) {hwAsymHtHF_ = hwAsymHtHF;}
    inline void setCentrality(int centrality) {centrality_ = centrality;}

    const int hwTotalEt() { return hwTotalEt_;}
    const int hwTotalEtEm() { return hwTotalEtEm_;}
    const int minBiasHFP0() { return minBiasHFP0_;}
    const int hwTotalHt() { return hwTotalHt_;}
    const int towerCount() { return towerCount_;}
    const int minBiasHFM0() { return minBiasHFM0_;}
    const int hwMissEt() { return hwMissEt_;}
    const int hwMissEtPhi() { return hwMissEtPhi_;}
    const int hwAsymEt() { return hwAsymEt_;}
    const int minBiasHFP1() { return minBiasHFP1_;}
    const int hwMissHt() { return hwMissHt_;}
    const int hwMissHtPhi() { return hwMissHtPhi_;}
    const int hwAsymHt() { return hwAsymHt_;}
    const int minBiasHFM1() { return minBiasHFM1_;}
    const int hwMissEtHF() { return hwMissEtHF_;}
    const int hwMissEtHFPhi() { return hwMissEtHFPhi_;}
    const int hwAsymEtHF() { return hwAsymEtHF_;}
    const int hwMissHtHF() { return hwMissHtHF_;}
    const int hwMissHtHFPhi() { return hwMissHtHFPhi_;}
    const int hwAsymHtHF() { return hwAsymHtHF_;}
    const int centrality() { return centrality_;}

  private:
    int hwTotalEt_, hwTotalEtEm_, minBiasHFP0_; // sums from ET block
    int hwTotalHt_, towerCount_, minBiasHFM0_; // sums from HT block
    int hwMissEt_, hwMissEtPhi_, hwAsymEt_, minBiasHFP1_; // sums from EtMiss block
    int hwMissHt_, hwMissHtPhi_, hwAsymHt_, minBiasHFM1_; // sums from HTMiss block
    int hwMissEtHF_, hwMissEtHFPhi_, hwAsymEtHF_; // sums from ETHFMiss block
    int hwMissHtHF_, hwMissHtHFPhi_, hwAsymHtHF_, centrality_; // sums from HTHFMiss block
  };

} // namespace l1ScoutingRun3
#endif // DataFormats_L1Scouting_L1ScoutingCalo_h
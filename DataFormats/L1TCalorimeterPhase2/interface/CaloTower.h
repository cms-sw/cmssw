// Description: Collection representing remaining ECAL energy post-L1EG clustering
// and the associated HCAL tower. The ECAL crystal TPs "simEcalEBTriggerPrimitiveDigis"
// within a 5x5 tower region are clustered to retain this ECAL energy info for passing
// to the GCT or further algos. The HCAL energy from the "simHcalTriggerPrimitiveDigis"
// is not specifically used in the L1EG algo beyond H/E, so the HCAL values here
// correspond to the full, initial HCAL TP energy.

#ifndef DataFormats_L1TCalorimeterPhase2_CaloTower_HH
#define DataFormats_L1TCalorimeterPhase2_CaloTower_HH

#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1tp2 {

  class CaloTower : public l1t::L1Candidate {
  public:
    CaloTower()
        : l1t::L1Candidate(),
          ecalTowerEt_(0.0),
          hcalTowerEt_(0.0),
          towerIPhi_(-99),
          towerIEta_(-99),
          towerPhi_(-99),
          towerEta_(-99),
          l1egTowerEt_(0.0),
          nL1eg_(0),
          l1egTrkSS_(0),
          l1egTrkIso_(0),
          l1egStandaloneSS_(0),
          l1egStandaloneIso_(0),
          isBarrel_(false){};

  public:
    inline float ecalTowerEt() const { return ecalTowerEt_; };
    inline float hcalTowerEt() const { return hcalTowerEt_; };
    inline int towerIPhi() const { return towerIPhi_; };
    inline int towerIEta() const { return towerIEta_; };
    inline float towerPhi() const { return towerPhi_; };
    inline float towerEta() const { return towerEta_; };
    inline float l1egTowerEt() const { return l1egTowerEt_; };
    inline int nL1eg() const { return nL1eg_; };
    inline int l1egTrkSS() const { return l1egTrkSS_; };
    inline int l1egTrkIso() const { return l1egTrkIso_; };
    inline int l1egStandaloneSS() const { return l1egStandaloneSS_; };
    inline int l1egStandaloneIso() const { return l1egStandaloneIso_; };
    inline bool isBarrel() const { return isBarrel_; };

    void setEcalTowerEt(float et) { ecalTowerEt_ = et; };
    void setHcalTowerEt(float et) { hcalTowerEt_ = et; };
    void setTowerIPhi(int iPhi) { towerIPhi_ = iPhi; };
    void setTowerIEta(int iEta) { towerIEta_ = iEta; };
    void setTowerPhi(float phi) { towerPhi_ = phi; };
    void setTowerEta(float eta) { towerEta_ = eta; };
    void setL1egTowerEt(float et) { l1egTowerEt_ = et; };
    void setNL1eg(int n) { nL1eg_ = n; };
    void setL1egTrkSS(int trkSS) { l1egTrkSS_ = trkSS; };
    void setL1egTrkIso(int trkIso) { l1egTrkIso_ = trkIso; };
    void setL1egStandaloneSS(int staSS) { l1egStandaloneSS_ = staSS; };
    void setL1egStandaloneIso(int staIso) { l1egStandaloneIso_ = staIso; };
    void setIsBarrel(bool isBarrel) { isBarrel_ = isBarrel; };

  private:
    float ecalTowerEt_ = 0.0;
    float hcalTowerEt_ = 0.0;
    int towerIPhi_ = -99;
    int towerIEta_ = -99;
    float towerPhi_ = -99;
    float towerEta_ = -99;

    // L1EG info
    float l1egTowerEt_ = 0.0;
    int nL1eg_ = 0;
    int l1egTrkSS_ = 0;
    int l1egTrkIso_ = 0;
    int l1egStandaloneSS_ = 0;
    int l1egStandaloneIso_ = 0;

    bool isBarrel_ = false;
  };

  // Collection of either ECAL or HCAL TPs with the Layer1 calibration constant attached, et_calibration
  typedef std::vector<CaloTower> CaloTowerCollection;

}  // namespace l1tp2
#endif

#ifndef DataFormats_L1TCalorimeter_HGCalTower_h
#define DataFormats_L1TCalorimeter_HGCalTower_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerID.h"

namespace l1t {

  class HGCalTower;
  typedef BXVector<HGCalTower> HGCalTowerBxCollection;

  class HGCalTower : public L1Candidate {
  public:
    HGCalTower() : etEm_(0.), etHad_(0.), id_(0), hwEtEm_(0), hwEtHad_(0), hwEtRatio_(0) {}

    HGCalTower(double etEm,
               double etHad,
               double eta,
               double phi,
               uint32_t id,
               int hwpt = 0,
               int hweta = 0,
               int hwphi = 0,
               int qual = 0,
               int hwEtEm = 0,
               int hwEtHad = 0,
               int hwEtRatio = 0);

    ~HGCalTower() override;

    void addEtEm(double et);
    void addEtHad(double et);

    double etEm() const { return etEm_; };
    double etHad() const { return etHad_; };

    const HGCalTower& operator+=(const HGCalTower& tower);

    HGCalTowerID id() const { return id_; }
    short zside() const { return id_.zside(); }

    void setHwEtEm(int et) { hwEtEm_ = et; }
    void setHwEtHad(int et) { hwEtHad_ = et; }
    void setHwEtRatio(int ratio) { hwEtRatio_ = ratio; }

    int hwEtEm() const { return hwEtEm_; }
    int hwEtHad() const { return hwEtHad_; }
    int hwEtRatio() const { return hwEtRatio_; }

  private:
    void addEt(double et);

    // additional hardware quantities
    double etEm_;
    double etHad_;
    HGCalTowerID id_;

    int hwEtEm_;
    int hwEtHad_;
    int hwEtRatio_;
  };

}  // namespace l1t

#endif

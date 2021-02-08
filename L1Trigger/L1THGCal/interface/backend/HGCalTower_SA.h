#ifndef L1Trigger_L1THGCal_HGCalTower_SA_h
#define L1Trigger_L1THGCal_HGCalTower_SA_h

namespace l1thgcfirmware {

  class HGCalTower {
  public:
    HGCalTower() {}
    HGCalTower(double etEm, double etHad) : etEm_(etEm), etHad_(etHad) {}

    ~HGCalTower(){};

    double etEm() const { return etEm_; };
    double etHad() const { return etHad_; };

    void addEtEm(double et);
    void addEtHad(double et);

    HGCalTower& operator+=(const HGCalTower& tower);

  private:
    double etEm_;
    double etHad_;
  };
}  // namespace l1thgcfirmware

#endif

#ifndef L1Trigger_L1THGCal_HGCalTower_SA_h
#define L1Trigger_L1THGCal_HGCalTower_SA_h

#include <cstdint>

namespace l1thgcfirmware {

  class HGCalTower {
  public:
    HGCalTower() = default;
    HGCalTower(double etEm, double etHad, float eta, float phi, uint32_t rawId)
        : etEm_(etEm), etHad_(etHad), eta_(eta), phi_(phi), id_(rawId) {}

    ~HGCalTower() = default;

    double etEm() const { return etEm_; }
    double etHad() const { return etHad_; }

    float eta() const { return eta_; }
    float phi() const { return phi_; }
    uint32_t id() const { return id_; }

    void addEtEm(double et);
    void addEtHad(double et);

    HGCalTower& operator+=(const HGCalTower& tower);

  private:
    double etEm_;
    double etHad_;

    float eta_;
    float phi_;
    uint32_t id_;
  };

  struct HGCalTowerCoord {
    HGCalTowerCoord(uint32_t rawId, float eta, float phi) : rawId(rawId), eta(eta), phi(phi) {}

    const uint32_t rawId;
    const float eta;
    const float phi;
  };
}  // namespace l1thgcfirmware

#endif

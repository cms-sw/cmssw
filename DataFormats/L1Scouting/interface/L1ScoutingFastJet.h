#ifndef DataFormats_L1Scouting_L1ScoutingFastJet_h
#define DataFormats_L1Scouting_L1ScoutingFastJet_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class FastJet {
  public:
    FastJet() : et_(0), eta_(0), phi_(0), nConst_(0), area_(0) {}

    FastJet(float et, float eta, float phi, int nConst, float area) : et_(et), eta_(eta), phi_(phi), nConst_(nConst), area_(area) {}

    void setEt(float et) { et_ = et; }
    void setEta(float eta) { eta_ = eta; }
    void setPhi(float phi) { phi_ = phi; }
    void setNConst(int nConst) { nConst_ = nConst; }
    void setArea(float area) { area_ = area; }

    float et() const { return et_; }
    float eta() const { return eta_; }
    float phi() const { return phi_; }
    int nConst() const { return nConst_; }
    float area() const { return area_; }

  private:
    float et_;
    float eta_;
    float phi_;
    int nConst_;
    float area_;
  };

  typedef OrbitCollection<FastJet> FastJetOrbitCollection;

}  // namespace l1ScoutingRun3
#endif  // DataFormats_L1Scouting_L1ScoutingFastJet_h

#ifndef DataFormats_L1Scouting_L1ScoutingCaloJet_h
#define DataFormats_L1Scouting_L1ScoutingCaloJet_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class CaloJet {
  public:
    CaloJet() : pt_(0), eta_(0), phi_(0), mass_(0), energyCorr_(1), nConst_(0) {}

    CaloJet(float pt, float eta, float phi, float mass, float energyCorr, int nConst)
        : pt_(pt), eta_(eta), phi_(phi), mass_(mass), energyCorr_(energyCorr), nConst_(nConst) {}

    void setPt(float pt) { pt_ = pt; }
    void setEta(float eta) { eta_ = eta; }
    void setPhi(float phi) { phi_ = phi; }
    void setMass(float mass) { mass_ = mass; }
    void setEnergyCorr(float energyCorr) { energyCorr_ = energyCorr; }
    void setNConst(int nConst) { nConst_ = nConst; }

    float pt() const { return pt_; }
    float eta() const { return eta_; }
    float phi() const { return phi_; }
    float mass() const { return mass_; }
    float energyCorr() const { return energyCorr_; }
    int nConst() const { return nConst_; }

  private:
    float pt_;
    float eta_;
    float phi_;
    float mass_;
    float energyCorr_;
    int nConst_;
  };

  typedef OrbitCollection<CaloJet> CaloJetOrbitCollection;

}  // namespace l1ScoutingRun3
#endif  // DataFormats_L1Scouting_L1ScoutingCaloJet_h

#ifndef DataFormats_L1TParticleFlow_PFTau_h
#define DataFormats_L1TParticleFlow_PFTau_h

#include <algorithm>
#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1t {

  static constexpr float PFTAU_NN_OFFSET = 0.1;
  static constexpr float PFTAU_NN_SLOPE = 0.2;
  static constexpr float PFTAU_NN_OVERALL_SCALE = 1. / 20.1;

  static constexpr float PFTAU_NN_LOOSE_CUT = 0.05;
  static constexpr float PFTAU_NN_TIGHT_CUT = 0.25;

  static constexpr float PFTAU_PF_LOOSE_CUT = 10.0;
  static constexpr float PFTAU_PF_TIGHT_CUT = 5.0;

  static constexpr double PFTAU_NN_PT_CUTOFF = 100.0;

  class PFTau : public L1Candidate {
  public:
    PFTau() {}
    enum { unidentified = 0, oneprong = 1, oneprongpi0 = 2, threeprong = 3 };
    PFTau(const LorentzVector& p,
          float iso = -1,
          float fulliso = -1,
          int id = 0,
          int hwpt = 0,
          int hweta = 0,
          int hwphi = 0)
        : PFTau(PolarLorentzVector(p), iso, id, hwpt, hweta, hwphi) {}
    PFTau(const PolarLorentzVector& p,
          float iso = -1,
          float fulliso = -1,
          int id = 0,
          int hwpt = 0,
          int hweta = 0,
          int hwphi = 0);
    float chargedIso() const { return iso_; }
    float fullIso() const { return fullIso_; }
    int id() const { return id_; }
    bool passLooseNN() const {
      return iso_ * (PFTAU_NN_OFFSET + PFTAU_NN_SLOPE * (min(pt(), PFTAU_NN_PT_CUTOFF))) * PFTAU_NN_OVERALL_SCALE >
             PFTAU_NN_LOOSE_CUT;
    }
    bool passLoosePF() const { return fullIso_ < PFTAU_PF_LOOSE_CUT; }
    bool passTightNN() const {
      return iso_ * (PFTAU_NN_OFFSET + PFTAU_NN_SLOPE * (min(pt(), PFTAU_NN_PT_CUTOFF))) * PFTAU_NN_OVERALL_SCALE >
             PFTAU_NN_TIGHT_CUT;
    }
    bool passTightPF() const { return fullIso_ < PFTAU_PF_TIGHT_CUT; }

  private:
    float iso_;
    float fullIso_;
    int id_;
  };

  typedef std::vector<l1t::PFTau> PFTauCollection;

  typedef edm::Ref<l1t::PFTauCollection> PFTauRef;
  typedef edm::RefVector<l1t::PFTauCollection> PFTauRefVector;
  typedef std::vector<l1t::PFTauRef> PFTauVectorRef;
}  // namespace l1t
#endif

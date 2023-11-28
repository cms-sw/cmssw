#ifndef DataFormats_L1TParticleFlow_PFTau_h
#define DataFormats_L1TParticleFlow_PFTau_h

#include <algorithm>
#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TParticleFlow/interface/taus.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"

namespace l1t {

  static constexpr float PFTAU_NN_OFFSET = 0.1;
  static constexpr float PFTAU_NN_SLOPE = 0.2;
  static constexpr float PFTAU_NN_OVERALL_SCALE = 1. / 20.1;

  static constexpr float PFTAU_NN_LOOSE_CUT = 0.28;
  static constexpr float PFTAU_NN_TIGHT_CUT = 0.25;

  static constexpr float PFTAU_PF_LOOSE_CUT = 10.0;
  static constexpr float PFTAU_PF_TIGHT_CUT = 5.0;

  static constexpr float PTSCALING_MASSCUT = 40.0;

  static constexpr double PFTAU_NN_PT_CUTOFF = 100.0;

  class PFTau : public L1Candidate {
  public:
    PFTau() {}
    enum { unidentified = 0, oneprong = 1, oneprongpi0 = 2, threeprong = 3 };
    PFTau(const LorentzVector& p,
          float iVector[80],
          float iso = -1,
          float fulliso = -1,
          int id = 0,
          int hwpt = 0,
          int hweta = 0,
          int hwphi = 0)
        : PFTau(PolarLorentzVector(p), iVector, iso, id, hwpt, hweta, hwphi) {}
    PFTau(const PolarLorentzVector& p,
          float iVector[80],
          float iso = -1,
          float fulliso = -1,
          int id = 0,
          int hwpt = 0,
          int hweta = 0,
          int hwphi = 0);
    float chargedIso() const { return iso_; }
    float fullIso() const { return fullIso_; }
    int id() const { return id_; }

    void setZ0(float z0) { setVertex(reco::Particle::Point(0, 0, z0)); }
    void setDxy(float dxy) { dxy_ = dxy; }

    float z0() const { return vz(); }
    float dxy() const { return dxy_; }
    const float* NNValues() const { return NNValues_; }

    bool passMass() const { return (mass() < 2 + pt() / PTSCALING_MASSCUT); }
    bool passLooseNN() const { return iso_ > PFTAU_NN_LOOSE_CUT; }
    bool passLooseNNMass() const {
      if (!passMass())
        return false;
      return passLooseNN();
    }
    bool passLoosePF() const { return fullIso_ < PFTAU_PF_LOOSE_CUT; }
    bool passTightNN() const {
      return iso_ * (PFTAU_NN_OFFSET + PFTAU_NN_SLOPE * (min(pt(), PFTAU_NN_PT_CUTOFF))) * PFTAU_NN_OVERALL_SCALE >
             PFTAU_NN_TIGHT_CUT;
    }
    bool passTightNNMass() const {
      if (!passMass())
        return false;
      return passTightNN();
    }
    bool passTightPF() const { return fullIso_ < PFTAU_PF_TIGHT_CUT; }

    //Tau encoding for GT
    void set_encodedTau(l1gt::PackedTau encodedTau) { encodedTau_ = encodedTau; }
    l1gt::PackedTau encodedTau() const { return encodedTau_; }  //Can be unpacked using l1gt::Tau::unpack()

    //Return the l1gt Tau object from the encoded objects
    l1gt::Tau getHWTauGT() const { return l1gt::Tau::unpack(encodedTau_); }

  private:
    float NNValues_[80];  // Values for each of the 80 NN inputs
    float iso_;
    float fullIso_;
    int id_;
    float dxy_;
    l1gt::PackedTau encodedTau_;
  };

  typedef std::vector<l1t::PFTau> PFTauCollection;

  typedef edm::Ref<l1t::PFTauCollection> PFTauRef;
  typedef edm::RefVector<l1t::PFTauCollection> PFTauRefVector;
  typedef std::vector<l1t::PFTauRef> PFTauVectorRef;
}  // namespace l1t
#endif

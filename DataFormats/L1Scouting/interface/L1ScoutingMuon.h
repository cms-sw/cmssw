#ifndef DataFormats_L1Scouting_L1ScoutingMuon_h
#define DataFormats_L1Scouting_L1ScoutingMuon_h

#include <cmath>

namespace scoutingRun3 {
  class ScMuon {
  public:

    ScMuon()
    : hwPt_(0),
      hwEta_(0),
      hwPhi_(0),
      hwQual_(0),
      hwChrg_(0),
      hwChrgv_(0),
      hwIso_(0),
      tfIndex_(0),
      hwEtaAtVtx_(0),
      hwPhiAtVtx_(0),
      hwPtUnconstrained_(0),
      hwDXY_(0) {}

    ScMuon(
      int hwPt,
      int hwEta,
      int hwPhi,
      int hwQual,
      int hwChrg,
      int hwChrgv,
      int hwIso,
      int tfIndex,
      int hwEtaAtVtx,
      int hwPhiAtVtx,
      int hwPtUnconstrained,
      int hwDXY)
    : hwPt_(hwPt),
      hwEta_(hwEta),
      hwPhi_(hwPhi),
      hwQual_(hwQual),
      hwChrg_(hwChrg),
      hwChrgv_(hwChrgv),
      hwIso_(hwIso),
      tfIndex_(tfIndex),
      hwEtaAtVtx_(hwEtaAtVtx),
      hwPhiAtVtx_(hwPhiAtVtx),
      hwPtUnconstrained_(hwPtUnconstrained),
      hwDXY_(hwDXY) {}

    ScMuon(const ScMuon& other) = default;
    ScMuon(ScMuon&& other) = default;
    ScMuon & operator=(const ScMuon& other) = default;
    ScMuon & operator=(ScMuon&& other) = default;

    void swap(ScMuon& other){
      using std::swap;
      swap(hwPt_, other.hwPt_);
      swap(hwEta_, other.hwEta_);
      swap(hwPhi_, other.hwPhi_);
      swap(hwQual_, other.hwQual_);
      swap(hwChrg_, other.hwChrg_);
      swap(hwChrgv_, other.hwChrgv_);
      swap(hwIso_, other.hwIso_);
      swap(tfIndex_, other.tfIndex_);
      swap(hwEtaAtVtx_, other.hwEtaAtVtx_);
      swap(hwPhiAtVtx_, other.hwPhiAtVtx_);
      swap(hwPtUnconstrained_, other.hwPtUnconstrained_);
      swap(hwDXY_, other.hwDXY_);
    }

    inline void setHwPt(int hwPt) { hwPt_= hwPt;}
    inline void setHwEta(int hwEta) { hwEta_= hwEta;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setHwQual(int hwQual) { hwQual_= hwQual;}
    inline void setHwChrg(int hwChrg) { hwChrg_= hwChrg;}
    inline void setHwChrgv(int hwChrgv) { hwChrgv_= hwChrgv;}
    inline void setHwIso(int hwIso) { hwIso_= hwIso;}
    inline void setTfIndex(int tfIndex) { tfIndex_= tfIndex;}
    inline void setHwEtaAtVtx(int hwEtaAtVtx) { hwEtaAtVtx_= hwEtaAtVtx;}
    inline void setHwPhiAtVtx(int hwPhiAtVtx) { hwPhiAtVtx_= hwPhiAtVtx;}
    inline void setHwPtUnconstrained(int hwPtUnconstrained) { hwPtUnconstrained_= hwPtUnconstrained;}
    inline void setHwDXY(int hwDXY) { hwDXY_= hwDXY;}

    inline int hwPt() const {return hwPt_;}
    inline int hwEta() const {return hwEta_;}
    inline int hwPhi() const {return hwPhi_;}
    inline int hwQual() const {return hwQual_;}
    inline int hwCharge() const {return hwChrg_;}
    inline int hwChargeValid() const {return hwChrgv_;}
    inline int hwIso() const {return hwIso_;}
    inline int hwIndex() const {return tfIndex_;}
    inline int hwEtaAtVtx() const {return hwEtaAtVtx_;}
    inline int hwPhiAtVtx() const {return hwPhiAtVtx_;}
    inline int hwPtUnconstrained() const {return hwPtUnconstrained_;}
    inline int hwDXY() const {return hwDXY_;}
    inline int tfMuonIndex() const {return tfIndex_;}
    
    // inline float pt() const {
    //   return pt_scale_*(hwPt_-1);
    // }
    // inline float eta()const {
    //   return eta_scale_*hwEta_;
    // }
    // inline float phi() const {
    //   return phi_scale_*hwPhi_;
    // }
    // inline float ptUnconstrained() const {
    //   return pt_scale_*(hwPtUnconstrained_-1);
    // }
    // inline float etaAtVtx() const {
    //   return eta_scale_*hwEtaAtVtx_;
    // }
    // inline float phiAtVtx() const {
    //   return phi_scale_*hwPhiAtVtx_;
    // }

  private:
    int hwPt_;
    int hwEta_;
    int hwPhi_;
    int hwQual_;
    int hwChrg_;
    int hwChrgv_;
    int hwIso_;
    int tfIndex_;
    int hwEtaAtVtx_;
    int hwPhiAtVtx_;
    int hwPtUnconstrained_;
    int hwDXY_;

    // constants to convert from harware to physical quantities
    // static constexpr float pt_scale_              = 0.5;
    // static constexpr float ptunconstrained_scale_ = 1.0;
    // static constexpr float phi_scale_             = 2.*M_PI/576.;
    // static constexpr float eta_scale_             = 0.0870/8;
  };

} // namespace scoutingRun3

#endif // DataFormats_L1Scouting_L1ScoutingMuon_h
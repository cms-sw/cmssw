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

    inline void setHwPt(int hwPt) { hwPt_= hwPt;}
    inline void setHwEta(int hwEta) { hwEta_= hwEta;}
    inline void setHwPhi(int hwPhi) { hwPhi_= hwPhi;}
    inline void setHwQual(int hwQual) { hwQual_= hwQual;}
    inline void setHwChrg(int hwChrg) { hwChrg_= hwChrg;}
    inline void setHwChrgv(int hwChrgv) { hwChrgv_= hwChrgv;}
    inline void setHwIso(int hwIso) { hwIso_= hwIso;}
    inline void setHfIndex(int tfIndex) { tfIndex_= tfIndex;}
    inline void setHwEtaAtVtx(int hwEtaAtVtx) { hwEtaAtVtx_= hwEtaAtVtx;}
    inline void setHwPhiAtVtx(int hwPhiAtVtx) { hwPhiAtVtx_= hwPhiAtVtx;}
    inline void setHwPtUnconstrained(int hwPtUnconstrained) { hwPtUnconstrained_= hwPtUnconstrained;}
    inline void setHwDXY(int hwDXY) { hwDXY_= hwDXY;}

    inline int getHwPt() const {return hwPt_;}
    inline int getHwEta() const {return hwEta_;}
    inline int getHwPhi() const {return hwPhi_;}
    inline int getHwQual() const {return hwQual_;}
    inline int getHwChrg() const {return hwChrg_;}
    inline int getHwChrgv() const {return hwChrgv_;}
    inline int getHwIso() const {return hwIso_;}
    inline int getHfIndex() const {return tfIndex_;}
    inline int getHwEtaAtVtx() const {return hwEtaAtVtx_;}
    inline int getHwPhiAtVtx() const {return hwPhiAtVtx_;}
    inline int getHwPtUnconstrained() const {return hwPtUnconstrained_;}
    inline int getHwDXY() const {return hwDXY_;}
    
    inline float getPt() const {
      return pt_scale_*(hwPt_-1);
    }
    inline float getEta()const {
      return eta_scale_*hwEta_;
    }
    inline float getPhi() const {
      return phi_scale_*hwPhi_;
    }
    inline float getPtUnconstrained() const {
      return pt_scale_*(hwPtUnconstrained_-1);
    }
    inline float getEtaAtVtx() const {
      return eta_scale_*hwEtaAtVtx_;
    }
    inline float getPhiAtVtx() const {
      return phi_scale_*hwPhiAtVtx_;
    }

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
    static constexpr float pt_scale_              = 0.5;
    static constexpr float ptunconstrained_scale_ = 1.0;
    static constexpr float phi_scale_             = 2.*M_PI/576.;
    static constexpr float eta_scale_             = 0.0870/8;
  };

} // namespace scoutingRun3
#endif // DataFormats_L1Scouting_L1ScoutingMuon_h
#ifndef DataFormats_L1Scouting_L1ScoutingMuon_h
#define DataFormats_L1Scouting_L1ScoutingMuon_h

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

namespace l1ScoutingRun3 {

  class Muon {
  public:
    Muon()
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

    Muon(int hwPt,
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

    void setHwPt(int hwPt) { hwPt_ = hwPt; }
    void setHwEta(int hwEta) { hwEta_ = hwEta; }
    void setHwPhi(int hwPhi) { hwPhi_ = hwPhi; }
    void setHwQual(int hwQual) { hwQual_ = hwQual; }
    void setHwChrg(int hwChrg) { hwChrg_ = hwChrg; }
    void setHwChrgv(int hwChrgv) { hwChrgv_ = hwChrgv; }
    void setHwIso(int hwIso) { hwIso_ = hwIso; }
    void setTfIndex(int tfIndex) { tfIndex_ = tfIndex; }
    void setHwEtaAtVtx(int hwEtaAtVtx) { hwEtaAtVtx_ = hwEtaAtVtx; }
    void setHwPhiAtVtx(int hwPhiAtVtx) { hwPhiAtVtx_ = hwPhiAtVtx; }
    void setHwPtUnconstrained(int hwPtUnconstrained) { hwPtUnconstrained_ = hwPtUnconstrained; }
    void setHwDXY(int hwDXY) { hwDXY_ = hwDXY; }

    int hwPt() const { return hwPt_; }
    int hwEta() const { return hwEta_; }
    int hwPhi() const { return hwPhi_; }
    int hwQual() const { return hwQual_; }
    int hwCharge() const { return hwChrg_; }
    int hwChargeValid() const { return hwChrgv_; }
    int hwIso() const { return hwIso_; }
    int hwIndex() const { return tfIndex_; }
    int hwEtaAtVtx() const { return hwEtaAtVtx_; }
    int hwPhiAtVtx() const { return hwPhiAtVtx_; }
    int hwPtUnconstrained() const { return hwPtUnconstrained_; }
    int hwDXY() const { return hwDXY_; }
    int tfMuonIndex() const { return tfIndex_; }

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
  };

  typedef OrbitCollection<Muon> MuonOrbitCollection;

}  // namespace l1ScoutingRun3

#endif  // DataFormats_L1Scouting_L1ScoutingMuon_h
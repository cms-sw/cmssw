#ifndef DataFormats_L1Trigger_Muon_h
#define DataFormats_L1Trigger_Muon_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

namespace l1t {

  class Muon;
  typedef BXVector<Muon> MuonBxCollection;
  typedef edm::Ref<MuonBxCollection> MuonRef;
  typedef edm::RefVector<MuonBxCollection> MuonRefVector;
  typedef std::vector<MuonRef> MuonVectorRef;

  typedef ObjectRefBxCollection<Muon> MuonRefBxCollection;
  typedef ObjectRefPair<Muon> MuonRefPair;
  typedef ObjectRefPairBxCollection<Muon> MuonRefPairBxCollection;

  class Muon : public L1Candidate {
  public:
    Muon();

    Muon(const LorentzVector& p4,
         int pt = 0,
         int eta = 0,
         int phi = 0,
         int qual = 0,
         int charge = 0,
         int chargeValid = 0,
         int iso = 0,
         int tfMuonIndex = -1,
         int tag = 0,
         bool debug = false,
         int isoSum = 0,
         int dPhi = 0,
         int dEta = 0,
         int rank = 0,
         int hwEtaAtVtx = 0,
         int hwPhiAtVtx = 0,
         double etaAtVtx = 0.,
         double phiAtVtx = 0.,
         int hwPtUnconstrained = 0,
         double ptUnconstrained = 0.,
         int dXY = 0);

    Muon(const PolarLorentzVector& p4,
         int pt = 0,
         int eta = 0,
         int phi = 0,
         int qual = 0,
         int charge = 0,
         int chargeValid = 0,
         int iso = 0,
         int tfMuonIndex = -1,
         int tag = 0,
         bool debug = false,
         int isoSum = 0,
         int dPhi = 0,
         int dEta = 0,
         int rank = 0,
         int hwEtaAtVtx = 0,
         int hwPhiAtVtx = 0,
         double etaAtVtx = 0.,
         double phiAtVtx = 0.,
         int hwPtUnconstrained = 0,
         double ptUnconstrained = 0.,
         int dXY = 0);

    ~Muon() override;

    // set values
    inline void setHwCharge(int charge) { hwCharge_ = charge; };
    inline void setHwChargeValid(int valid) { hwChargeValid_ = valid; };
    inline void setTfMuonIndex(int index) { tfMuonIndex_ = index; };
    inline void setHwTag(int tag) { hwTag_ = tag; };

    inline void setHwEtaAtVtx(int hwEtaAtVtx) { hwEtaAtVtx_ = hwEtaAtVtx; };
    inline void setHwPhiAtVtx(int hwPhiAtVtx) { hwPhiAtVtx_ = hwPhiAtVtx; };
    inline void setEtaAtVtx(double etaAtVtx) { etaAtVtx_ = etaAtVtx; };
    inline void setPhiAtVtx(double phiAtVtx) { phiAtVtx_ = phiAtVtx; };

    inline void setHwIsoSum(int isoSum) { hwIsoSum_ = isoSum; };
    inline void setHwDPhiExtra(int dPhi) { hwDPhiExtra_ = dPhi; };
    inline void setHwDEtaExtra(int dEta) { hwDEtaExtra_ = dEta; };
    inline void setHwRank(int rank) { hwRank_ = rank; };

    inline void setHwPtUnconstrained(int hwPtUnconstrained) { hwPtUnconstrained_ = hwPtUnconstrained; };
    inline void setPtUnconstrained(double ptUnconstrained) { ptUnconstrained_ = ptUnconstrained; };
    inline void setHwDXY(int hwDXY) { hwDXY_ = hwDXY; };

    inline void setDebug(bool debug) { debug_ = debug; };

    // methods to retrieve values
    inline int hwCharge() const { return hwCharge_; };
    inline int hwChargeValid() const { return hwChargeValid_; };
    inline int tfMuonIndex() const { return tfMuonIndex_; };
    inline int hwTag() const { return hwTag_; };

    inline int hwEtaAtVtx() const { return hwEtaAtVtx_; };
    inline int hwPhiAtVtx() const { return hwPhiAtVtx_; };
    inline double etaAtVtx() const { return etaAtVtx_; };
    inline double phiAtVtx() const { return phiAtVtx_; };

    inline int hwIsoSum() const { return hwIsoSum_; };
    inline int hwDPhiExtra() const { return hwDPhiExtra_; };
    inline int hwDEtaExtra() const { return hwDEtaExtra_; };
    inline int hwRank() const { return hwRank_; };

    inline int hwPtUnconstrained() const { return hwPtUnconstrained_; };
    inline double ptUnconstrained() const { return ptUnconstrained_; };
    inline int hwDXY() const { return hwDXY_; };

    inline bool debug() const { return debug_; };

    virtual bool operator==(const l1t::Muon& rhs) const;
    virtual inline bool operator!=(const l1t::Muon& rhs) const { return !(operator==(rhs)); };

  private:
    // additional hardware quantities common to L1 global jet
    int hwCharge_;
    int hwChargeValid_;
    int tfMuonIndex_;
    int hwTag_;

    // additional hardware quantities only available if debug flag is set
    bool debug_;
    int hwIsoSum_;
    int hwDPhiExtra_;
    int hwDEtaExtra_;
    int hwRank_;

    // muon coordinates at the vertex
    int hwEtaAtVtx_;
    int hwPhiAtVtx_;
    double etaAtVtx_;
    double phiAtVtx_;

    // displacement information
    int hwPtUnconstrained_;
    double ptUnconstrained_;
    int hwDXY_;
  };

}  // namespace l1t

#endif

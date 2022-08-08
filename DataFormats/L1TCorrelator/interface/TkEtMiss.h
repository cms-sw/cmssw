#ifndef DataFormatsL1TCorrelator_TkEtMiss_h
#define DataFormatsL1TCorrelator_TkEtMiss_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

namespace l1t {
  class TkEtMiss : public L1Candidate {
  public:
    enum EtMissType { kMET, kMHT, kNumTypes };
    TkEtMiss();
    TkEtMiss(const LorentzVector& p4,
             EtMissType type,
             const double& etTotal,
             const double& etMissPU,
             const double& etTotalPU,
             const edm::Ref<l1t::VertexWordCollection>& aVtxRef = edm::Ref<l1t::VertexWordCollection>(),
             int bx = 0);

    TkEtMiss(const LorentzVector& p4,
             EtMissType type,
             const double& etTotal,
             const double& etMissPU,
             const double& etTotalPU,
             int bx = 0);

    TkEtMiss(const LorentzVector& p4, EtMissType type, const double& EtPhi, const int& NumTracks, int bx = 0);

    // ---------- const member functions ---------------------
    EtMissType type() const { return type_; }  // kMET or kMHT
    // For type = kMET, this is |MET|; for type = kMHT, this is |MHT|
    double etMiss() const { return et(); }
    // For type = kMET, this is total ET; for type = kMHT, this is total HT
    double etTotal() const { return etTot_; }
    // EtMiss and EtTot from PU vertices
    double etMissPU() const { return etMissPU_; }
    double etTotalPU() const { return etTotalPU_; }
    int bx() const { return bx_; }
    const edm::Ref<l1t::VertexWordCollection>& vtxRef() const { return vtxRef_; }

    double etPhi() const { return etPhi_; }
    int etQual() const { return etQual_; }

    // ---------- member functions ---------------------------
    void setEtTotal(const double& etTotal) { etTot_ = etTotal; }
    void setBx(int bx) { bx_ = bx; }

  private:
    // ---------- member data --------------------------------
    EtMissType type_;
    double etTot_;
    double etMissPU_;
    double etTotalPU_;
    edm::Ref<l1t::VertexWordCollection> vtxRef_;

    double etMiss_;
    double etPhi_;
    int etQual_;

    int bx_;
  };
}  // namespace l1t

#endif

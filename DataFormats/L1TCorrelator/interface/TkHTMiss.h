#ifndef DataFormatsL1TCorrelator_TkHTMiss_h
#define DataFormatsL1TCorrelator_TkHTMiss_h
// Package:     L1Trigger
// Class  :     TkHTMiss

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"

namespace l1t {
  class TkHTMiss : public L1Candidate {
  public:
    TkHTMiss();
    TkHTMiss(const LorentzVector& p4,
             double EtTotal,
             const edm::RefProd<TkJetCollection>& jetCollRef = edm::RefProd<TkJetCollection>(),
             const edm::Ref<TkPrimaryVertexCollection>& aVtxRef = edm::Ref<TkPrimaryVertexCollection>(),
             int bx = 0);

    // ---------- const member functions ---------------------
    double EtMiss() const {  // HTM (missing HT)
      return et();
    }
    double etTotal() const { return etTot_; }
    // HTM and HT from PU vertices
    double etMissPU() const { return etMissPU_; }
    double etTotalPU() const { return etTotalPU_; }
    int bx() const { return bx_; }
    float vtx() const { return zvtx_; }
    const edm::RefProd<TkJetCollection>& jetCollectionRef() const { return jetCollectionRef_; }
    const edm::Ref<TkPrimaryVertexCollection>& vtxRef() const { return vtxRef_; }

    // ---------- member functions ---------------------------
    void setEtTotal(double EtTotal) { etTot_ = EtTotal; }
    void setEtTotalPU(double EtTotalPU) { etTotalPU_ = EtTotalPU; }
    void setEtMissPU(double EtMissPU) { etMissPU_ = EtMissPU; }
    void setVtx(const float& zvtx) { zvtx_ = zvtx; }
    void setBx(int bx) { bx_ = bx; }

  private:
    // ---------- member data --------------------------------
    float zvtx_;        // zvtx used to constrain the jets
    double etTot_;      // HT
    double etMissPU_;   // HTM form jets that don't come from zvtx
    double etTotalPU_;  // HT from jets that don't come from zvtx

    edm::RefProd<TkJetCollection> jetCollectionRef_;
    edm::Ref<TkPrimaryVertexCollection> vtxRef_;

    int bx_;
  };
}  // namespace l1t

#endif

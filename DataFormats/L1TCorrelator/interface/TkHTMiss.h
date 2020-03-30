#ifndef TkTrigger_TkHTMiss_h
#define TkTrigger_TkHTMiss_h
// Package:     L1Trigger
// Class  :     TkHTMiss
// Original Author:  E. Perez
//         Created:  Nov 14, 2013

// system include files
// user include files
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
    TkHTMiss(
        const LorentzVector& p4,
        const double& EtTotal,
        const edm::RefProd<TkJetCollection>& jetCollRef = edm::RefProd<TkJetCollection>(),
        const edm::Ref<TkPrimaryVertexCollection>& aVtxRef = edm::Ref<TkPrimaryVertexCollection>(),
        int bx = 0);

    // ---------- const member functions ---------------------
    double EtMiss() const {  // HTM (missing HT)
      return et();
    }
    const double& EtTotal() const { return EtTot_; }
    // HTM and HT from PU vertices
    double EtMissPU() const { return EtMissPU_; }
    double EtTotalPU() const { return EtTotalPU_; }
    int bx() const { return bx_; }
    float vtx() const { return zvtx_; }
    const edm::RefProd<TkJetCollection>& jetCollectionRef() const { return jetCollectionRef_; }
    const edm::Ref<TkPrimaryVertexCollection>& vtxRef() const { return vtxRef_; }

    // ---------- member functions ---------------------------
    void setEtTotal(const double& EtTotal) { EtTot_ = EtTotal; }
    void setEtTotalPU(const double& EtTotalPU) { EtTotalPU_ = EtTotalPU; }
    void setEtMissPU(const double& EtMissPU) { EtMissPU_ = EtMissPU; }
    void setVtx(const float& zvtx) { zvtx_ = zvtx; }
    void setBx(int bx) { bx_ = bx; }

  private:
    // ---------- member data --------------------------------
    float zvtx_;        // zvtx used to constrain the jets
    double EtTot_;      // HT
    double EtMissPU_;   // HTM form jets that don't come from zvtx
    double EtTotalPU_;  // HT from jets that don't come from zvtx

    edm::RefProd<TkJetCollection> jetCollectionRef_;
    edm::Ref<TkPrimaryVertexCollection> vtxRef_;

    int bx_;
  };
}  // namespace l1t

#endif

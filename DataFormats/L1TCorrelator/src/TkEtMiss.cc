#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"

using namespace l1t;

TkEtMiss::TkEtMiss() {}

TkEtMiss::TkEtMiss(const LorentzVector& p4,
                   EtMissType type,
                   const double& etTotal,
                   const double& etMissPU,
                   const double& etTotalPU,
                   const edm::Ref<TkPrimaryVertexCollection>& avtxRef,
                   int bx)
    : L1Candidate(p4),
      type_(type),
      etTot_(etTotal),
      etMissPU_(etMissPU),
      etTotalPU_(etTotalPU),
      vtxRef_(avtxRef),
      bx_(bx) {}

TkEtMiss::TkEtMiss(const LorentzVector& p4,
                   EtMissType type,
                   const double& etTotal,
                   const double& etMissPU,
                   const double& etTotalPU,
                   int bx)
    : L1Candidate(p4), type_(type), etTot_(etTotal), etMissPU_(etMissPU), etTotalPU_(etTotalPU), bx_(bx) {}

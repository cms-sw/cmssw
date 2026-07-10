// F.R.
#ifndef JetReco_CaloMETfwd_h
#define JetReco_CaloMETfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  namespace io_v1 {
    class CaloMET;
  }
  using CaloMET = io_v1::CaloMET;
  /// collection of CaloMET objects
  typedef std::vector<CaloMET> CaloMETCollection;
  /// edm references
  typedef edm::Ref<CaloMETCollection> CaloMETRef;
  typedef edm::RefVector<CaloMETCollection> CaloMETRefVector;
  typedef edm::RefProd<CaloMETCollection> CaloMETRefProd;
}  // namespace reco
#endif

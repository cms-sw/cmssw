// F.R.
#ifndef JetReco_CaloMETfwd_h
#define JetReco_CaloMETfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class CaloMET;
  /// collection of CaloMET objects 
  typedef std::vector<CaloMET> CaloMETCollection;
  /// edm references
  typedef edm::Ref<CaloMETCollection> CaloMETRef;
  typedef edm::RefVector<CaloMETCollection> CaloMETRefVector;
  typedef edm::RefProd<CaloMETCollection> CaloMETRefProd;
}
#endif

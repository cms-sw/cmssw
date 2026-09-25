// F.R.
#ifndef JetReco_GenMETfwd_h
#define JetReco_GenMETfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  namespace io_v1 {
    class GenMET;
  }
  using io_v1::GenMET;
  /// collection of GenMET objects
  typedef std::vector<GenMET> GenMETCollection;
  /// edm references
  typedef edm::Ref<GenMETCollection> GenMETRef;
  typedef edm::RefVector<GenMETCollection> GenMETRefVector;
  typedef edm::RefProd<GenMETCollection> GenMETRefProd;
}  // namespace reco
#endif

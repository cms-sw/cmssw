// F.R.
#ifndef JetReco_METfwd_h
#define JetReco_METfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class MET;
  /// collection of MET objects 
  typedef std::vector<MET> METCollection;
  /// edm references
  typedef edm::Ref<METCollection> METRef;
  typedef edm::RefVector<METCollection> METRefVector;
  typedef edm::RefProd<METCollection> METRefProd;
}
#endif

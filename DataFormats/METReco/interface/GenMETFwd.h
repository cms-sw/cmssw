// F.R.
// $Id: GenMETFwd.h,v 1.1 2006/08/04 05:39:28 cavana Exp $
#ifndef JetReco_GenMETfwd_h
#define JetReco_GenMETfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class GenMET;
  /// collection of GenMET objects 
  typedef std::vector<GenMET> GenMETCollection;
  /// edm references
  typedef edm::Ref<GenMETCollection> GenMETRef;
  typedef edm::RefVector<GenMETCollection> GenMETRefVector;
  typedef edm::RefProd<GenMETCollection> GenMETRefProd;
}
#endif

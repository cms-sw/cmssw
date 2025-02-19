// author: R. Remington
// date: 10/27/08

#ifndef METReco_PFMETfwd_h
#define METReco_PFMETfwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFMET;
  /// collection of PFMET objects 
  typedef std::vector<PFMET> PFMETCollection;
  /// edm references
  typedef edm::Ref<PFMETCollection> PFMETRef;
  typedef edm::RefVector<PFMETCollection> PFMETRefVector;
  typedef edm::RefProd<PFMETCollection> PFMETRefProd;
}
#endif



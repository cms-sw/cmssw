// author: Salvatore Rappoccio
// date: 28-Dec-2010

#ifndef METReco_PFClusterMETfwd_h
#define METReco_PFClusterMETfwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFClusterMET;
  /// collection of PFClusterMET objects
  typedef std::vector<PFClusterMET> PFClusterMETCollection;
  /// edm references
  typedef edm::Ref<PFClusterMETCollection> PFClusterMETRef;
  typedef edm::RefVector<PFClusterMETCollection> PFClusterMETRefVector;
  typedef edm::RefProd<PFClusterMETCollection> PFClusterMETRefProd;
}  // namespace reco
#endif

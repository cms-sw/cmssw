#ifndef EgammaReco_PreshowerClusterFwd_h
#define EgammaReco_PreshowerClusterFwd_h
//
// $Id: PreshowerClusterFwd.h,v 1.4 2006/06/11 17:59:41 rahatlou Exp $
//
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace reco {
  class PreshowerCluster;

  /// collection of PreshowerCluster objects
  typedef std::vector<PreshowerCluster> PreshowerClusterCollection;

  /// persistent reference to PreshowerCluster objects
  typedef edm::Ref<PreshowerClusterCollection> PreshowerClusterRef;

  /// reference to PreshowerCluster collection
  typedef edm::RefProd<PreshowerClusterCollection> PreshowerClusterRefProd;

  /// vector of references to PreshowerCluster objects all in the same collection
  typedef edm::RefVector<PreshowerClusterCollection> PreshowerClusterRefVector;

  /// iterator over a vector of references to PreshowerCluster objects
  typedef PreshowerClusterRefVector::iterator PreshowerCluster_iterator;
}

#endif

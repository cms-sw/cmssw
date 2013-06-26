#ifndef EgammaReco_ClusterShapeFwd_h
#define EgammaReco_ClusterShapeFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ClusterShape;

  /// collection of ClusterShape objects
  typedef std::vector<ClusterShape> ClusterShapeCollection;

  /// reference to an object in a collection of ClusterShape objects
  typedef edm::Ref<ClusterShapeCollection> ClusterShapeRef;

  /// reference to a collection of ClusterShape objects
  typedef edm::RefProd<ClusterShapeCollection> ClusterShapeRefProd;

  /// vector of references to objects in the same collectin of ClusterShape objects
  typedef edm::RefVector<ClusterShapeCollection> ClusterShapeRefVector;

  /// iterator over a vector of references to ClusterShape objects
  typedef ClusterShapeRefVector::iterator clusterShape_iterator;
}

#endif

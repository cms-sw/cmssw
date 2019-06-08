#ifndef EgammaReco_PreshowerClusterShapeShapeFwd_h
#define EgammaReco_PreshowerClusterShapeShapeFwd_h
//
// author Aris Kyriakis (NCSR "Demokritos")
//
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace reco {
  class PreshowerClusterShape;

  /// collection of PreshowerClusterShape objects
  typedef std::vector<PreshowerClusterShape> PreshowerClusterShapeCollection;

  /// persistent reference to PreshowerClusterShape objects
  typedef edm::Ref<PreshowerClusterShapeCollection> PreshowerClusterShapeRef;

  /// reference to PreshowerClusterShape collection
  typedef edm::RefProd<PreshowerClusterShapeCollection> PreshowerClusterShapeRefProd;

  /// vector of references to PreshowerClusterShape objects all in the same collection
  typedef edm::RefVector<PreshowerClusterShapeCollection> PreshowerClusterShapeRefVector;

  /// iterator over a vector of references to PreshowerClusterShape objects
  typedef PreshowerClusterShapeRefVector::iterator PreshowerClusterShape_iterator;
}  // namespace reco

#endif

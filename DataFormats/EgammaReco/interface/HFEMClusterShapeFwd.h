#ifndef HFEMClusterShapeFwd_h
#define HFEMClusterShapeFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class HFEMClusterShape;
// collection of HFEMClusterShape objects
typedef std::vector<HFEMClusterShape> HFEMClusterShapeCollection;

/// persistent reference to HFEMClusterShape objects
typedef edm::Ref<HFEMClusterShapeCollection> HFEMClusterShapeRef;

/// reference to HFEMClusterShape collection
typedef edm::RefProd<HFEMClusterShapeCollection> HFEMClusterShapeRefProd;

/// vector of references to HFEMClusterShape objects all in the same collection
typedef edm::RefVector<HFEMClusterShapeCollection> HFEMClusterShapeRefVector;

/// iterator over a vector of references to HFEMClusterShape objects
typedef HFEMClusterShapeRefVector::iterator HFEMClusterShape_iterator;
 
}

#endif

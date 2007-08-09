
#ifndef HFEMClusterShapeAssociation_h
#define HFEMClusterShapeAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<SuperClusterCollection, HFEMClusterShapeCollection> > HFEMClusterShapeAssociationCollection;
 
  typedef HFEMClusterShapeAssociationCollection::value_type HFEMClusterShapeAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<HFEMClusterShapeAssociationCollection> HFEMClusterShapeAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<HFEMClusterShapeAssociationCollection> HFEMClusterShapeAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<HFEMClusterShapeAssociationCollection> HFEMClusterShapeAssociationRefVector;

}

#endif

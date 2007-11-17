#ifndef SeedSuperClusterAssociation_h
#define SeedSuperClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<TrajectorySeedCollection, SuperClusterCollection> > SeedSuperClusterAssociationCollection;
 
  typedef SeedSuperClusterAssociationCollection::value_type SeedSuperClusterAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<SeedSuperClusterAssociationCollection> SeedSuperClusterAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<SeedSuperClusterAssociationCollection> SeedSuperClusterAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<SeedSuperClusterAssociationCollection> SeedSuperClusterAssociationRefVector;

}

#endif

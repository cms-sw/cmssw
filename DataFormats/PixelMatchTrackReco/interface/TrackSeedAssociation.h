#ifndef TrackSeedAssociation_h
#define TrackSeedAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<TrackCollection,TrajectorySeedCollection> > TrackSeedAssociationCollection;
 
  typedef TrackSeedAssociationCollection::value_type TrackSeedAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<TrackSeedAssociationCollection> TrackSeedAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<TrackSeedAssociationCollection> TrackSeedAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<TrackSeedAssociationCollection> TrackSeedAssociationRefVector;

}

#endif

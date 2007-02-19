#ifndef TrackSuperClusterAssociation_h
#define TrackSuperClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"


namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<TrackCollection,SuperClusterCollection> > TrackSuperClusterAssociationCollection;
 

  typedef TrackSuperClusterAssociationCollection::value_type TrackSuperClusterAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<TrackSuperClusterAssociationCollection> TrackSuperClusterAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<TrackSuperClusterAssociationCollection> TrackSuperClusterAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<TrackSuperClusterAssociationCollection> TrackSuperClusterAssociationRefVector;

}

#endif

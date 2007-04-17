#ifndef GsfTrackSeedAssociation_h
#define GsfTrackSeedAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<GsfTrackCollection,TrajectorySeedCollection> > GsfTrackSeedAssociationCollection;
 
  typedef GsfTrackSeedAssociationCollection::value_type GsfTrackSeedAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<GsfTrackSeedAssociationCollection> GsfTrackSeedAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<GsfTrackSeedAssociationCollection> GsfTrackSeedAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<GsfTrackSeedAssociationCollection> GsfTrackSeedAssociationRefVector;

}

#endif

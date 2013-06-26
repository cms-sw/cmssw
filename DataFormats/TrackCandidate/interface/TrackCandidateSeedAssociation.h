#ifndef TrackCandidateSeedAssociation_h
#define TrackCandidateSeedAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<TrackCandidateCollection,TrajectorySeedCollection> > TrackCandidateSeedAssociationCollection;
 
  typedef TrackCandidateSeedAssociationCollection::value_type TrackCandidateSeedAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<TrackCandidateSeedAssociationCollection> TrackCandidateSeedAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<TrackCandidateSeedAssociationCollection> TrackCandidateSeedAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<TrackCandidateSeedAssociationCollection> TrackCandidateSeedAssociationRefVector;

}

#endif

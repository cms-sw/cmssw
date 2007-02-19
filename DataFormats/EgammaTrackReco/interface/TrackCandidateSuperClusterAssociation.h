#ifndef TrackCandidateSuperClusterAssociation_h
#define TrackCandidateSuperClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"



namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<TrackCandidateCollection,SuperClusterCollection> > TrackCandidateSuperClusterAssociationCollection;
 

  typedef TrackCandidateSuperClusterAssociationCollection::value_type TrackCandidateSuperClusterAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<TrackCandidateSuperClusterAssociationCollection> TrackCandidateSuperClusterAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<TrackCandidateSuperClusterAssociationCollection> TrackCandidateSuperClusterAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<TrackCandidateSuperClusterAssociationCollection> TrackCandidateSuperClusterAssociationRefVector;

}

#endif

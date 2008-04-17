#ifndef PhotonIDAssociation_h
#define PhotonIDAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<PhotonCollection, PhotonIDCollection> > PhotonIDAssociationCollection;
 
  typedef PhotonIDAssociationCollection::value_type PhotonIDAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<PhotonIDAssociationCollection> PhotonIDAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<PhotonIDAssociationCollection> PhotonIDAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<PhotonIDAssociationCollection> PhotonIDAssociationRefVector;

}

#endif

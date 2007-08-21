#ifndef ParticleFlowReco_PFClusterShapeAssociation_h_
#define ParticleFlowReco_PFClusterShapeAssociation_h_
//  \class PFClusterShapeAssociation
//
//  \association of ClusterShape to PFCluster
//
//
//
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace reco 
{
  typedef
    edm::AssociationMap<edm::OneToOne<PFClusterCollection, ClusterShapeCollection> >
    PFClusterShapeAssociationCollection;
  typedef
    PFClusterShapeAssociationCollection::value_type PFClusterShapeAssociation;
  typedef
    edm::Ref<PFClusterShapeAssociationCollection> PFClusterShapeAssociationRef;
  typedef
    edm::RefProd<PFClusterShapeAssociationCollection> PFClusterShapeAssociationRefProd;
  typedef
    edm::RefVector<PFClusterShapeAssociationCollection> PFClusterShapeAssociationRefVector;  
}

#endif

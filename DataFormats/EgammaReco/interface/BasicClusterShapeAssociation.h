#ifndef EgammaReco_BasicClusterShapeAssociation_h
#define EgammaReco_BasicClusterShapeAssociation_h
//  \class BasicClusterShapeAssociation
//
//  \association of ClusterShape to BasicCluster
//
//
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace reco {
  typedef 
    edm::AssociationMap<edm::OneToOne<BasicClusterCollection, ClusterShapeCollection> > 
    BasicClusterShapeAssociationCollection;
  typedef 
    BasicClusterShapeAssociationCollection::value_type BasicClusterShapeAssociation;
  typedef 
    edm::Ref<BasicClusterShapeAssociationCollection> BasicClusterShapeAssociationRef;
  typedef 
    edm::RefProd<BasicClusterShapeAssociationCollection> BasicClusterShapeAssociationRefProd;
  typedef 
    edm::RefVector<BasicClusterShapeAssociationCollection> BasicClusterShapeAssociationRefVector;  
}

#endif

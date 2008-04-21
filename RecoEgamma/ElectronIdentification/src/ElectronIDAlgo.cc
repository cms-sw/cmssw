#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"

void ElectronIDAlgo::baseSetup(const edm::ParameterSet& conf) {
  barrelClusterShapeAssocProducer_ = conf.getParameter<edm::InputTag>("barrelClusterShapeAssociation");
  endcapClusterShapeAssocProducer_ = conf.getParameter<edm::InputTag>("endcapClusterShapeAssociation");
}

const reco::ClusterShape& ElectronIDAlgo::getClusterShape(const reco::GsfElectron* electron, const edm::Event& e) {

  // Get association maps linking BasicClusters to ClusterShape.
  edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleBarrel ;
  e.getByLabel(barrelClusterShapeAssocProducer_, clusterShapeHandleBarrel) ;
  if (!clusterShapeHandleBarrel.isValid()) {
    edm::LogError ("ElectronIDAlgo") << "Can't get ECAL barrel Cluster Shape Collection" ; 
  }
  const reco::BasicClusterShapeAssociationCollection& barrelClShpMap = *clusterShapeHandleBarrel ;
  edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleEndcap ;
  e.getByLabel(endcapClusterShapeAssocProducer_, clusterShapeHandleEndcap) ;
  if (!clusterShapeHandleEndcap.isValid()) {
    edm::LogError ("ElectronIDAlgo") << "Can't get ECAL endcap Cluster Shape Collection" ; 
  }
  const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *clusterShapeHandleEndcap ;
  
  // Find entry in map corresponding to seed BasicCluster of SuperCluster
  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;

  if ( electron->classification() < 100 ) {
    reco::SuperClusterRef sclusRef = electron->get<reco::SuperClusterRef> () ;
    seedShpItr = barrelClShpMap.find ( sclusRef->seed () ) ;
    if ( seedShpItr == barrelClShpMap.end () )
         seedShpItr = endcapClShpMap.find ( sclusRef->seed ()) ;
    }
  else {
    reco::SuperClusterRef sclusRef = electron->get<reco::SuperClusterRef> () ;
    seedShpItr = endcapClShpMap.find ( sclusRef->seed () ) ;
    }

   const reco::ClusterShapeRef& sClShape = seedShpItr->val ;
   return (*sClShape) ;

}

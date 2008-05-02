#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"

ElectronIDSelectorLikelihood::ElectronIDSelectorLikelihood (const edm::ParameterSet& conf) : conf_ (conf) 
{
  doLikelihood_ = conf_.getParameter<bool> ("doLikelihood"); 
  barrelClusterShapeAssociation_ = conf_.getParameter<edm::InputTag>("barrelClusterShapeAssociation");
  endcapClusterShapeAssociation_ = conf_.getParameter<edm::InputTag>("endcapClusterShapeAssociation");

}

ElectronIDSelectorLikelihood::~ElectronIDSelectorLikelihood () 
{
}

void ElectronIDSelectorLikelihood::newEvent (const edm::Event& e, const edm::EventSetup& c)
{
  
  c.getData(likelihoodAlgo_) ;

}

double ElectronIDSelectorLikelihood::operator () (const reco::GsfElectron & electron, const edm::Event& event) 
{

  // get the association of the clusters to their shapes for EB
  edm::Handle<reco::BasicClusterShapeAssociationCollection> barrelClShpHandle ;
  event.getByLabel (barrelClusterShapeAssociation_, barrelClShpHandle) ;
  if (!barrelClShpHandle.isValid()) {
    edm::LogError ("ElectronIDProducer") << "Can't get ECAL barrel Cluster Shape Collection" ; 
  }
  const reco::BasicClusterShapeAssociationCollection& barrelClShpMap = *barrelClShpHandle ;

  // get the association of the clusters to their shapes for EE
  edm::Handle<reco::BasicClusterShapeAssociationCollection> endcapClShpHandle ;
  event.getByLabel (endcapClusterShapeAssociation_, endcapClShpHandle) ;
  if (!endcapClShpHandle.isValid()) {
    edm::LogError ("ElectronIDProducer") << "Can't get ECAL endcap Cluster Shape Collection" ;
  }
  const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *endcapClShpHandle ;

  if (doLikelihood_) 
  {
    bool hasBarrel=true ;
    bool hasEndcap=true ;

    reco::SuperClusterRef sclusRef = electron.get<reco::SuperClusterRef> () ;
    reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr ;
    seedShpItr = barrelClShpMap.find (sclusRef->seed ()) ;
    if ( seedShpItr==barrelClShpMap.end ())  {
      hasBarrel=false ;
      seedShpItr=endcapClShpMap.find (sclusRef->seed ()) ;
      if ( seedShpItr==endcapClShpMap.end () ) hasEndcap=false ;
    }
    if (hasBarrel || hasEndcap) {
      const reco::ClusterShapeRef& sClShape = seedShpItr->val ;
      return static_cast<double>(likelihoodAlgo_->result (electron,*sClShape)) ;
    }
  }
  return 0. ;

}

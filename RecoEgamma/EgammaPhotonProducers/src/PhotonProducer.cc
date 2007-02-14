#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"


PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  // use onfiguration file to setup input/output collection names
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");

  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  barrelClusterShapeMapProducer_   = conf_.getParameter<std::string>("barrelClusterShapeMapProducer");
  barrelClusterShapeMapCollection_ = conf_.getParameter<std::string>("barrelClusterShapeMapCollection");
  endcapClusterShapeMapProducer_   = conf_.getParameter<std::string>("endcapClusterShapeMapProducer");
  endcapClusterShapeMapCollection_ = conf_.getParameter<std::string>("endcapClusterShapeMapCollection");
  vertexProducer_       = conf_.getParameter<std::string>("primaryVertexProducer");
  PhotonCollection_ = conf_.getParameter<std::string>("photonCollection");

  // Register the product
  produces< reco::PhotonCollection >(PhotonCollection_);

}

PhotonProducer::~PhotonProducer() {

}


void  PhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {


}


void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  //
  // create empty output collections
  //

  reco::PhotonCollection outputPhotonCollection;
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection_p(new reco::PhotonCollection);

  // Get the  Barrel Super Cluster collection
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);

  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  edm::LogInfo("PhotonProducer") << " Accessing Barrel SC collection with size : " << scBarrelCollection.size()  << "\n";

 // Get the  Endcap Super Cluster collection
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);

  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  edm::LogInfo("PhotonProducer") << " Accessing Endcap SC collection with size : " << scEndcapCollection.size()  << "\n";
  
  // Get ClusterShape association maps
  edm::Handle<reco::BasicClusterShapeAssociationCollection> barrelClShpHandle;
  theEvent.getByLabel(barrelClusterShapeMapProducer_, barrelClusterShapeMapCollection_, barrelClShpHandle);
  const reco::BasicClusterShapeAssociationCollection& barrelClShpMap = *barrelClShpHandle;

  edm::Handle<reco::BasicClusterShapeAssociationCollection> endcapClShpHandle;
  theEvent.getByLabel(endcapClusterShapeMapProducer_, endcapClusterShapeMapCollection_, endcapClShpHandle);
  const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *endcapClShpHandle;

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  if (vertexProducer_ != "") {
    theEvent.getByLabel(vertexProducer_, vertexHandle);
    vertexCollection = *(vertexHandle.product());
  }
  math::XYZPoint vtx(0.,0.,0.);
  if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();

  edm::LogInfo("PhotonProducer") << "Constructing Photon 4-vectors assuming primary vertex position: " << vtx << std::endl;

  int iSC=0; // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  fillPhotonCollection(scBarrelHandle,barrelClShpMap,vtx,outputPhotonCollection,iSC);
  fillPhotonCollection(scEndcapHandle,endcapClShpMap,vtx,outputPhotonCollection,iSC);

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  theEvent.put( outputPhotonCollection_p, PhotonCollection_);

}

void PhotonProducer::fillPhotonCollection(
		   const edm::Handle<reco::SuperClusterCollection> & scHandle,
		   const reco::BasicClusterShapeAssociationCollection& clshpMap,
		   math::XYZPoint & vtx,
		   reco::PhotonCollection & outputPhotonCollection, int iSC) {

  reco::SuperClusterCollection scCollection = *(scHandle.product());
  reco::SuperClusterCollection::iterator aClus;
  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;

  std::cout << "h1" << std::endl;
  int lSC=0; // reset local supercluster index
  for(aClus = scCollection.begin(); aClus != scCollection.end(); aClus++) {

    std::cout << "h2" << std::endl;
    // compute R9=E3x3/ESC
    seedShpItr = clshpMap.find(aClus->seed());
    assert(seedShpItr != clshpMap.end());
    const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
    double r9 = seedShapeRef->e3x3()/(aClus->rawEnergy()+aClus->preshowerEnergy());
    double r19 = seedShapeRef->eMax()/seedShapeRef->e3x3();
    double e5x5 = seedShapeRef->e5x5();

    // compute position of ECAL shower
    math::XYZPoint caloPosition;
    caloPosition = aClus->position();

    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::Photon newCandidate(0, p4, r9, r19, e5x5, vtx);

    outputPhotonCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scHandle, lSC));
    outputPhotonCollection[iSC].setSuperCluster(scRef);
 
    iSC++;
    lSC++;

  }

}

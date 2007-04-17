#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"


PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  // use onfiguration file to setup input/output collection names
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");

  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
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


  //  Loop over barrel SC and fill the  photon collection
  int iSC=0; // index in photon collection
  int lSC=0; // local index on barrel
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); aClus++) {

    const reco::Particle::Point  vtx( 0, 0, 0 );

    // compute correctly the momentum vector of the photon from primary vertex and cluster position
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::Photon newCandidate(0, p4, vtx);

    outputPhotonCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
    outputPhotonCollection[iSC].setSuperCluster(scRef);


    lSC++;
    iSC++;

  }

  //  Loop over Endcap SC and fill the  photon collection
  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); aClus++) {

    const reco::Particle::Point  vtx( 0, 0, 0 );

    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::Photon newCandidate(0, p4, vtx);

    outputPhotonCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
    outputPhotonCollection[iSC].setSuperCluster(scRef);
 
    iSC++;
    lSC++;

  }

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  theEvent.put( outputPhotonCollection_p, PhotonCollection_);

}

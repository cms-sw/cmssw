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

//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
// Class header file
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"


PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  std::cout << " PhotonProducer CTOR " << std::endl;

  // use onfiguration file to setup input/output collection names
 

  scProducer_       = conf_.getParameter<std::string>("scProducer");
  scCollection_     = conf_.getParameter<std::string>("scCollection");
  PhotonCollection_ = conf_.getParameter<std::string>("photonCollection");

  // Register the product
  produces< reco::PhotonCollection >(PhotonCollection_);



}

PhotonProducer::~PhotonProducer() {


}


void  PhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {

  //get magnetic field
  edm::LogInfo("PhotonProducer") << "get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  


}


void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  edm::LogInfo("PhotonProducer") << "Analyzing event number: " << theEvent.id() << "\n";


  //
  // create empty output collections
  //

  reco::PhotonCollection outputPhotonCollection;
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection_p(new reco::PhotonCollection);
  std::cout << " Created empty PhotonCollection size " <<   std::endl;




  // Get the Super Cluster collection
  Handle<reco::SuperClusterCollection> scHandle;
  try{  
    theEvent.getByLabel(scProducer_,scCollection_,scHandle);
  } catch ( cms::Exception& ex ) {
    LogError("PhotonProducer") << "Error! can't get the SC " << scCollection_.c_str() ;
  } 
  std::cout << " Trying to access SC collection from my Producer " << std::endl;
  reco::SuperClusterCollection scCollection = *(scHandle.product());
  std::cout << " SC collection size  " << scCollection.size() << std::endl;


  //  Loop over SC and fill the  photon collection
  int iSC=0;
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scCollection.begin(); aClus != scCollection.end(); aClus++) {

    const reco::Particle::LorentzVector  p4(0, 0, 0, aClus->energy() );
    const reco::Particle::Point  vtx( 0, 0, 0 );
    reco::Photon newCandidate(0, p4, vtx);

    outputPhotonCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scHandle, iSC));
    outputPhotonCollection[iSC].setSuperCluster(scRef);

    iSC++;      

      
  }
  
   

  // put the product in the event
  std::cout << " Put the PhotonCollection " << iSC << "  candidates " << std::endl;
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  theEvent.put( outputPhotonCollection_p, PhotonCollection_);


}

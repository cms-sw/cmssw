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
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonCorrectionProducer.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgo.h"

PhotonCorrectionProducer::PhotonCorrectionProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  std::cout << " PhotonCorrectionProducer CTOR " << std::endl;

  // use onfiguration file to setup input/output collection names
 

  phoProducer_       = conf_.getParameter<std::string>("phoProducer");
  phoCollection_     = conf_.getParameter<std::string>("phoCollection");

  PhotonCollection_ = conf_.getParameter<std::string>("photonCorrCollection");

  // Register the product
  produces< reco::PhotonCollection >(PhotonCollection_);

  // switch on/off corrections
  applyDummyCorrection_=conf_.getParameter<bool>("applyDummyCorrection");


}

PhotonCorrectionProducer::~PhotonCorrectionProducer() {

  delete theDummyCorrection_;

}


void  PhotonCorrectionProducer::beginJob (edm::EventSetup const & theEventSetup) {

  //get magnetic field
  edm::LogInfo("PhotonCorrectionProducer") << "get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  
  


  theDummyCorrection_= new  PhotonCorrectionAlgo();


}


void PhotonCorrectionProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  edm::LogInfo("PhotonCorrectionProducer") << "Producing event number: " << theEvent.id() << "\n";


  //
  // create empty output collections
  //
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection(new reco::PhotonCollection);
  std::cout << " Created empty uncorrected PhotonCollection " <<   std::endl;


  // Get the uncorrected photon Collection

  Handle<reco::PhotonCollection> phoHandle;
  try{  
    theEvent.getByLabel(phoProducer_,phoCollection_,phoHandle);
  } catch ( cms::Exception& ex ) {
    LogError("PhotonCorrectionProducer") << "Error! can't get the Uncorrected Photon " << phoCollection_.c_str() ;
  } 
  std::cout << " Trying to access the Uncorrected photon collection from my Producer " << std::endl;
  reco::PhotonCollection phoCollection = *(phoHandle.product());
  std::cout << " Uncorrected Photon collection size  " << phoCollection.size() << std::endl;


  //  Loop over the uncorrected photons and attach the corrections
  int myCands=0;
  reco::PhotonCollection::iterator iPho;
  for(iPho = phoCollection.begin(); iPho != phoCollection.end(); iPho++) {

    //     reco::Photon newCandidate;
      if( applyDummyCorrection_) theDummyCorrection_->makeCorrections(&(*iPho));
      outputPhotonCollection->push_back(*iPho);
      myCands++;      

      
  }
  
    



  // put the product in the event
  std::cout << " Put the PhotonCollection " << myCands << "  corrected candidates " << std::endl;
  theEvent.put( outputPhotonCollection, PhotonCollection_);


}

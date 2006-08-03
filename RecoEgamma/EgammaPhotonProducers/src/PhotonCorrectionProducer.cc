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
// Class header file
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonCorrectionProducer.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonDummyCorrection.h"

PhotonCorrectionProducer::PhotonCorrectionProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  edm::LogInfo(" PhotonCorrectionProducer CTOR ") << "\n";

  // use onfiguration file to setup input/output collection names
 

  photonCorrectionProducer_         = conf_.getParameter<std::string>("photonCorrectionProducer");
  uncorrectedPhotonCollection_     = conf_.getParameter<std::string>("uncorrectedPhotonCollection");

  CorrectedPhotonCollection_ = conf_.getParameter<std::string>("correctedPhotonCollection");

  // Register the product
  produces< reco::PhotonCollection >(CorrectedPhotonCollection_);

  // switch on/off corrections
  applyDummyCorrection_=conf_.getParameter<bool>("applyDummyCorrection");


}

PhotonCorrectionProducer::~PhotonCorrectionProducer() {

  delete theDummyCorrection_;

}


void  PhotonCorrectionProducer::beginJob (edm::EventSetup const & theEventSetup) {


  theDummyCorrection_= new  PhotonDummyCorrection();


}


void PhotonCorrectionProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  //
  // create empty output collections
  //
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection(new reco::PhotonCollection);



  // Get the uncorrected photon Collection

  Handle<reco::PhotonCollection> uncorrectedPhotonHandle;
  theEvent.getByLabel(photonCorrectionProducer_,uncorrectedPhotonCollection_,uncorrectedPhotonHandle);
  reco::PhotonCollection phoCollection = *(uncorrectedPhotonHandle.product());
  LogInfo("PhotonCorrectionProducer: Uncorrected Photon collection size ") << phoCollection.size() << "\n";


  //  Loop over the uncorrected photons and attach the corrections
  int myCands=0;
  reco::PhotonCollection::iterator iPho;
  for(iPho = phoCollection.begin(); iPho != phoCollection.end(); iPho++) {

    //     reco::Photon newCandidate;
    if( applyDummyCorrection_) { 
      LogInfo("PhotonCorrectionProducer: Applying dummy correction")  << "\n";
         theDummyCorrection_->makeCorrections( *iPho );
    }
      outputPhotonCollection->push_back(*iPho);
      myCands++;
  }

  // put the product in the event
  LogInfo("Put in the event ") << myCands << "  corrected candidates " << "\n";
  theEvent.put( outputPhotonCollection, CorrectedPhotonCollection_);

}

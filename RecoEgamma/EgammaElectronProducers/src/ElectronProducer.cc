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
#include "DataFormats/EgammaCandidates/interface/Electron.h"

//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
// Class header file
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronProducer.h"


ElectronProducer::ElectronProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  std::cout << " ElectronProducer CTOR " << std::endl;

  // use onfiguration file to setup input/output collection names
 

  scProducer_       = conf_.getParameter<std::string>("scProducer");
  scCollection_     = conf_.getParameter<std::string>("scCollection");
  ElectronCollection_ = conf_.getParameter<std::string>("electronCollection");

  // Register the product
  produces< reco::ElectronCollection >(ElectronCollection_);



}

ElectronProducer::~ElectronProducer() {


}


void  ElectronProducer::beginJob (edm::EventSetup const & theEventSetup) {

  //get magnetic field
  edm::LogInfo("ElectronProducer") << "get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  


}


void ElectronProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  edm::LogInfo("ElectronProducer") << "Analyzing event number: " << theEvent.id() << "\n";


  //
  // create empty output collections
  //

  reco::ElectronCollection outputElectronCollection;
  std::auto_ptr< reco::ElectronCollection > outputElectronCollection_p(new reco::ElectronCollection);





  // Get the Super Cluster collection
  Handle<reco::SuperClusterCollection> scHandle;
  try{  
    theEvent.getByLabel(scProducer_,scCollection_,scHandle);
  } catch ( cms::Exception& ex ) {
    LogError("ElectronProducer") << "Error! can't get the SC " << scCollection_.c_str() ;
  } 
 
  reco::SuperClusterCollection scCollection = *(scHandle.product());
 


  //  Loop over SC and fill the  photon collection
  int iSC=0;
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scCollection.begin(); aClus != scCollection.end(); aClus++) {

    const reco::Particle::LorentzVector  p4(0, 0, 0, aClus->energy() );
    const reco::Particle::Point  vtx( 0, 0, 0 );
    reco::Electron newCandidate(0, p4, vtx);

    outputElectronCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scHandle, iSC));
    outputElectronCollection[iSC].setSuperCluster(scRef);

    iSC++;      

      
  }
  
   

  // put the product in the event
  std::cout << " Put the ElectronCollection " << iSC << "  candidates " << std::endl;
  outputElectronCollection_p->assign(outputElectronCollection.begin(),outputElectronCollection.end());
  theEvent.put( outputElectronCollection_p, ElectronCollection_);


}

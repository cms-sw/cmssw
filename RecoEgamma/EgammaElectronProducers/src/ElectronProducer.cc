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
 

  scBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");

  scBarrelCollection_     = conf_.getParameter<std::string>("scBarrelCollection");
  scEndcapCollection_     = conf_.getParameter<std::string>("scEndcapCollection");
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

  edm::LogInfo("ElectronProducer") << "Producing event number: " << theEvent.id() << "\n";


  //
  // create empty output collections
  //

  reco::ElectronCollection outputElectronCollection;
  std::auto_ptr< reco::ElectronCollection > outputElectronCollection_p(new reco::ElectronCollection);
  std::cout << " Created empty ElectronCollection size " <<   std::endl;




  // Get the  Barrel Super Cluster collection
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scBarrelProducer_,scBarrelCollection_,scBarrelHandle);
  std::cout << " Trying to access barrel SC collection from my Producer " << std::endl;
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  std::cout << " barrel SC collection size  " << scBarrelCollection.size() << std::endl;

 // Get the  Endcap Super Cluster collection
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scEndcapProducer_,scEndcapCollection_,scEndcapHandle);
  std::cout << " Trying to access endcap SC collection from my Producer " << std::endl;
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  std::cout << " endcap SC collection size  " << scEndcapCollection.size() << std::endl;



  //  Loop over barrel SC and fill the  photon collection
  int iSC=0;
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); aClus++) {

    const reco::Particle::Point  vtx( 0, 0, 0 );

    // compute correctly the momentum vector of the photon from primary vertex and cluster position
    math::XYZVector direction = aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::Electron newCandidate(-1, p4, vtx);

    outputElectronCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, iSC));
    outputElectronCollection[iSC].setSuperCluster(scRef);

    iSC++;

  }

  //  Loop over Endcap SC and fill the  photon collection
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); aClus++) {

    const reco::Particle::LorentzVector  p4(0, 0, 0, aClus->energy() );
    const reco::Particle::Point  vtx( 0, 0, 0 );
    reco::Electron newCandidate(0, p4, vtx);

    outputElectronCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, iSC));
    outputElectronCollection[iSC].setSuperCluster(scRef);

    iSC++;      

      
  }
  
   

  // put the product in the event
  std::cout << " Put the ElectronCollection " << iSC << "  candidates " << std::endl;
  outputElectronCollection_p->assign(outputElectronCollection.begin(),outputElectronCollection.end());
  theEvent.put( outputElectronCollection_p, ElectronCollection_);


}

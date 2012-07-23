// -*- C++ -*-
//
//

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include <iostream>

#include "Calibration/EcalCalibAlgos/interface/ElectronRecalibSuperClusterAssociator.h"
//#define DEBUG 

using namespace reco;
using namespace edm;
 
ElectronRecalibSuperClusterAssociator::ElectronRecalibSuperClusterAssociator(const edm::ParameterSet& iConfig) 
{
#ifdef DEBUG
  std::cout<< "ElectronRecalibSuperClusterAssociator::ElectronRecalibSuperClusterAssociator" << std::endl;
#endif

  //register your products
  produces<GsfElectronCollection>();
  produces<GsfElectronCoreCollection>() ;
  //  produces<SuperClusterCollection>();
  
  scProducer_ = iConfig.getParameter<std::string>("scProducer");
  scCollection_ = iConfig.getParameter<std::string>("scCollection");

  scIslandProducer_ = iConfig.getParameter<std::string>("scIslandProducer");
  scIslandCollection_ = iConfig.getParameter<std::string>("scIslandCollection");

  electronProducer_ = iConfig.getParameter<std::string > ("electronProducer");
  electronCollection_ = iConfig.getParameter<std::string > ("electronCollection");
#ifdef DEBUG
  std::cout<< "ElectronRecalibSuperClusterAssociator::ElectronRecalibSuperClusterAssociator::end" << std::endl;
#endif
}

ElectronRecalibSuperClusterAssociator::~ElectronRecalibSuperClusterAssociator()
{
}

// ------------ method called to produce the data  ------------
void ElectronRecalibSuperClusterAssociator::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
#ifdef DEBUG
  std::cout<< "ElectronRecalibSuperClusterAssociator::produce" << std::endl;
#endif
  // Create the output collections   
  std::auto_ptr<GsfElectronCollection> pOutEle(new GsfElectronCollection);
  std::auto_ptr<GsfElectronCoreCollection> pOutEleCore(new GsfElectronCoreCollection);
  //  std::auto_ptr<SuperClusterCollection> pOutNewEndcapSC(new SuperClusterCollection);

//   reco::SuperClusterRefProd rSC = e.getRefBeforePut<SuperClusterCollection>();
//   edm::Ref<SuperClusterCollection>::key_type idxSC = 0;

  //Get Hybrid SuperClusters
  Handle<reco::SuperClusterCollection> pSuperClusters;
  e.getByLabel(scProducer_, scCollection_, pSuperClusters);
  if (!pSuperClusters.isValid()) {
    std::cerr << "Error! can't get the product SuperClusterCollection "<< std::endl;
  }
  const reco::SuperClusterCollection* scCollection = pSuperClusters.product();
  
#ifdef DEBUG
  std::cout<<"scCollection->size()"<<scCollection->size()<<std::endl;
#endif
  
  //Get Island SuperClusters
  Handle<reco::SuperClusterCollection> pIslandSuperClusters;
  e.getByLabel(scIslandProducer_, scIslandCollection_, pIslandSuperClusters);
  if (!pIslandSuperClusters.isValid()) {
    std::cerr << "Error! can't get the product IslandSuperClusterCollection "<< std::endl;
  }
  const reco::SuperClusterCollection* scIslandCollection = pIslandSuperClusters.product();
  
#ifdef DEBUG
  std::cout<<"scEECollection->size()"<<scIslandCollection->size()<<std::endl;
#endif

  // Get Electrons
  Handle<edm::View<reco::GsfElectron> > pElectrons;
  e.getByLabel(electronProducer_, electronCollection_, pElectrons);
  if (!pElectrons.isValid()) {
    std::cerr << "Error! can't get the product ElectronCollection "<< std::endl;
  }
  const edm::View<reco::GsfElectron>* electronCollection = pElectrons.product();

  GsfElectronCoreRefProd rEleCore=e.getRefBeforePut<GsfElectronCoreCollection>();
  edm::Ref<GsfElectronCoreCollection>::key_type idxEleCore = 0;
  
  for(edm::View<reco::GsfElectron>::const_iterator eleIt = electronCollection->begin(); eleIt != electronCollection->end(); eleIt++)
    {
      float DeltaRMineleSCbarrel(0.15); 
      float DeltaRMineleSCendcap(0.15); 
      const reco::SuperCluster* nearestSCbarrel=0;
      const reco::SuperCluster* nearestSCendcap=0;
      int iscRef=-1;
      int iSC=0;
      
      // first loop is on EB superClusters
      for(reco::SuperClusterCollection::const_iterator scIt = scCollection->begin();
	  scIt != scCollection->end(); scIt++){
#ifdef DEBUG	
	std::cout << scIt->energy() << " " << scIt->eta() << " " << scIt->phi() << " " << eleIt->eta() << " " << eleIt->phi() << std::endl;
#endif
	
	double DeltaReleSC = sqrt ( pow(  eleIt->eta() - scIt->eta(),2) + pow(eleIt->phi() - scIt->phi(),2));
	
	if(DeltaReleSC<DeltaRMineleSCbarrel)
	  {
	    DeltaRMineleSCbarrel = DeltaReleSC;
	    nearestSCbarrel = &*scIt;
	    iscRef = iSC;
	  }
	iSC++;
      }
      iSC = 0;
      
      // second loop is on EE superClusters
      int iscRefendcap=-1;
      
      for(reco::SuperClusterCollection::const_iterator scItEndcap = scIslandCollection->begin();
	  scItEndcap != scIslandCollection->end(); scItEndcap++){
#ifdef DEBUG	
	std::cout << "EE " << scItEndcap->energy() << " " << scItEndcap->eta() << " " << scItEndcap->phi() << " " << eleIt->eta() << " " << eleIt->phi() << std::endl;
#endif
	
	double DeltaReleSC = sqrt ( pow(  eleIt->eta() - scItEndcap->eta(),2) + pow(eleIt->phi() - scItEndcap->phi(),2));
	
	if(DeltaReleSC<DeltaRMineleSCendcap)
	  {
	    DeltaRMineleSCendcap = DeltaReleSC;
	    nearestSCendcap = &*scItEndcap;
	    iscRefendcap = iSC;
	  }
	iSC++;
      }
      ////////////////////////      



#ifdef DEBUG
      std::cout << "+++++++++++" << std::endl;
      std::cout << &(*eleIt->gsfTrack()) << std::endl;
      std::cout << eleIt->core()->ecalDrivenSeed() << std::endl;
      std::cout << "+++++++++++" << std::endl;
#endif
      if(eleIt->isEB() && nearestSCbarrel){
#ifdef DEBUG
	std::cout << "Starting Association is with EB superCluster "<< std::endl;
#endif  
 	reco::GsfElectronCore newEleCore(*(eleIt->core()));
	newEleCore.setGsfTrack(eleIt->gsfTrack());
	reco::SuperClusterRef scRef(reco::SuperClusterRef(pSuperClusters, iscRef));
	newEleCore.setSuperCluster(scRef);
	reco::GsfElectronCoreRef newEleCoreRef(reco::GsfElectronCoreRef(rEleCore, idxEleCore ++));
	pOutEleCore->push_back(newEleCore);
	reco::GsfElectron newEle(*eleIt,newEleCoreRef);
	//,CaloClusterPtr(),
	//				  TrackRef(),GsfTrackRefVector());
	//TrackRef(),TrackBaseRef(), GsfTrackRefVector());
	newEle.setCorrectedEcalEnergy(eleIt->p4().energy()*(nearestSCbarrel->energy()/eleIt->ecalEnergy()));
	//	std::cout << "FROM REF " << newEle.superCluster().key() << std::endl;
	pOutEle->push_back(newEle);
#ifdef DEBUG
	std::cout << "Association is with EB superCluster "<< std::endl;
#endif  
      }  

      if(!(eleIt->isEB()) && nearestSCendcap)
	{
#ifdef DEBUG
	std::cout << "Starting Association is with EE superCluster "<< std::endl;
#endif  

// 	float preshowerEnergy=eleIt->superCluster()->preshowerEnergy(); 
// #ifdef DEBUG
// 	  std::cout << "preshowerEnergy"<< preshowerEnergy << std::endl;
// #endif
// 	  /// fixme : should have a vector of ptr of ref, to avoid copying
// 	  CaloClusterPtrVector newBCRef;
// 	  for (CaloCluster_iterator bcRefIt=nearestSCendcap->clustersBegin();bcRefIt!=nearestSCendcap->clustersEnd();++bcRefIt){
// 	    CaloClusterPtr cPtr(*bcRefIt);
// 	    newBCRef.push_back(cPtr);
// 	  }
	 

// 	  //	  reco::SuperCluster newSC(nearestSCendcap->energy() + preshowerEnergy, nearestSCendcap->position() , nearestSCendcap->seed(),newBCRef , preshowerEnergy );
// 	  reco::SuperCluster newSC(nearestSCendcap->energy(), nearestSCendcap->position() , nearestSCendcap->seed(),newBCRef , preshowerEnergy );
// 	  pOutNewEndcapSC->push_back(newSC);
// 	  reco::SuperClusterRef scRef(reco::SuperClusterRef(rSC, idxSC ++));
	  
	  reco::GsfElectronCore newEleCore(*(eleIt->core()));
	  newEleCore.setGsfTrack(eleIt->gsfTrack());
	  reco::SuperClusterRef scRef(reco::SuperClusterRef(pIslandSuperClusters, iscRefendcap));
	  newEleCore.setSuperCluster(scRef);
	  reco::GsfElectronCoreRef newEleCoreRef(reco::GsfElectronCoreRef(rEleCore, idxEleCore ++));
	  pOutEleCore->push_back(newEleCore);
	  reco::GsfElectron newEle(*eleIt,newEleCoreRef);
	  //,CaloClusterPtr(),
	  //				  TrackRef(),GsfTrackRefVector());
	  //TrackRef(),TrackBaseRef(), GsfTrackRefVector());
	  newEle.setCorrectedEcalEnergy(eleIt->p4().energy()*(nearestSCendcap->energy()/eleIt->ecalEnergy()));
	//	std::cout << "FROM REF " << newEle.superCluster().key() << std::endl;
	  pOutEle->push_back(newEle);
// 	  reco::GsfElectronCore newEleCore(*(eleIt->core()));
// 	  newEleCore.setGsfTrack(eleIt->gsfTrack());
// 	  newEleCore.setSuperCluster(scRef);
// 	  reco::GsfElectronCoreRef newEleCoreRef(reco::GsfElectronCoreRef(rEleCore, idxEleCore ++));
// 	  pOutEleCore->push_back(newEleCore);
// 	  reco::GsfElectron newEle(*eleIt,newEleCoreRef,CaloClusterPtr(),
// //				  TrackRef(),GsfTrackRefVector());
// 				  TrackRef(),TrackBaseRef(), GsfTrackRefVector());
// 	  newEle.setCorrectedEcalEnergy(eleIt->p4().energy()*(newSC.energy()/eleIt->ecalEnergy()),eleIt->ecalEnergyError()*(newSC.energy()/eleIt->ecalEnergy())); 
// 	  pOutEle->push_back(newEle);

#ifdef DEBUG
	std::cout << "Association is with EE superCluster "<< std::endl;
#endif  
      }  
    
    }
  
  
  
#ifdef DEBUG
  std::cout << "Filled new electrons  " << pOutEle->size() << std::endl;
  std::cout << "Filled new electronsCore  " << pOutEleCore->size() << std::endl;
  //  std::cout << "Filled new endcapSC  " << pOutNewEndcapSC->size() << std::endl;
#endif  
  
  // put result into the Event

  e.put(pOutEle);
  e.put(pOutEleCore);
  //  e.put(pOutNewEndcapSC);
  
}

DEFINE_FWK_MODULE(ElectronRecalibSuperClusterAssociator);

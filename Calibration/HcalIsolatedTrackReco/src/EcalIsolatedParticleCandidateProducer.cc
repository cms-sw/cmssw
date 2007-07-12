// -*- C++ -*-
//
// Package:    EcalIsolatedParticleCandidateProducer
// Class:      EcalIsolatedParticleCandidateProducer
// 
/**\class EcalIsolatedParticleCandidateProducer EcalIsolatedParticleCandidateProducer.cc Calibration/EcalIsolatedParticleCandidateProducer/src/EcalIsolatedParticleCandidateProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory Safronov
//         Created:  Thu Jun  7 17:21:58 MSD 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Calibration/HcalIsolatedTrackReco/interface/EcalIsolatedParticleCandidateProducer.h"
#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidateFwd.h"

#include "FWCore/Framework/interface/produce_helpers.h"


EcalIsolatedParticleCandidateProducer::EcalIsolatedParticleCandidateProducer(const edm::ParameterSet& conf)
{
  useEndcap_=conf.getParameter<bool>("UseEndcap");
  coneSize_ = conf.getParameter<double>("EcalIsolationConeSize");
  minEnergy_= conf.getParameter<double>("MinEnergy");
  barrelBclusterProducer_=conf.getParameter<std::string>("BarrelBasicClusterProducer");
  endcapBclusterProducer_=conf.getParameter<std::string>("EndcapBasicClusterProducer");
  barrelBclusterCollectionLabel_=conf.getParameter<std::string>("BarrelBasicClusterCollectionLabel");
  endcapBclusterCollectionLabel_=conf.getParameter<std::string>("EndcapBasicClusterCollectionLabel");


   //register your products
   produces< reco::EcalIsolatedParticleCandidateCollection >();

}


EcalIsolatedParticleCandidateProducer::~EcalIsolatedParticleCandidateProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalIsolatedParticleCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::BasicClusterCollection> basCluEB;
  iEvent.getByLabel(barrelBclusterProducer_,barrelBclusterCollectionLabel_, basCluEB);
   
  edm::Handle<reco::BasicClusterCollection> basCluEE;
  if (useEndcap_) iEvent.getByLabel(endcapBclusterProducer_, endcapBclusterCollectionLabel_,basCluEE);
    
  reco::EcalIsolatedParticleCandidateCollection * eipcCollection=new reco::EcalIsolatedParticleCandidateCollection;

  //  edm::LogInfo("AAAAAAAAAA")<<"supCluEB size: "<<supCluEB->size()<<"\n"<<"supCluEE size: "<<supCluEE->size()<<"\n"<<"basCluEB size: "<<basCluEB->size()<<"\n"<<"basCluEB size: "<<basCluEE->size()<<" BasEB id: "<<rrr->id()<<"Bas EE id: "<<rrr1->id();

  //  double scCount;

  //      for (unsigned int i=0; i<basCluEB->size(); i++)
  for (reco::BasicClusterCollection::const_iterator it_b=basCluEB->begin(); it_b!=basCluEB->end(); it_b++)	
    {
      if (it_b->energy()<minEnergy_) continue;
      double eta=it_b->eta();
      double phi=it_b->phi();
      double energy=it_b->energy();
      double maxNear=-10;
      double sumNear=0;
      for (reco::BasicClusterCollection::const_iterator it_in=basCluEB->begin(); it_in!=basCluEB->end(); it_in++)	
	{   
	  double dPhi;
	  if (it_in->phi()-it_b->phi()>3.14159) dPhi=6.28319-(it_in->phi()-it_b->phi());
	  else dPhi=it_in->phi()-it_b->phi();
	  if (it_in==it_b||(sqrt(pow(it_in->eta()-it_b->eta(),2)+pow(dPhi,2))>coneSize_)) continue;
	  if (maxNear<it_in->energy()) maxNear=it_in->energy();
	  sumNear+=it_in->energy();
	}
      if (useEndcap_) 
	{
	  for (reco::BasicClusterCollection::const_iterator it_in=basCluEE->begin(); it_in!=basCluEE->end(); it_in++)
	    {
	      double dPhi;
	      if (it_in->phi()-it_b->phi()>3.14159) dPhi=6.28319-(it_in->phi()-it_b->phi());
	      else dPhi=it_in->phi()-it_b->phi();
	      if (it_in==it_b||(sqrt(pow(it_in->eta()-it_b->eta(),2)+pow(dPhi,2))>coneSize_)) continue;
	      if (maxNear<it_in->energy()) maxNear=it_in->energy();
	      sumNear+=it_in->energy();
	    }
	}
      
       reco::EcalIsolatedParticleCandidate newca(eta, phi, energy, maxNear, sumNear);
       eipcCollection->push_back(newca);
    }
  if (useEndcap_)
    {
      for (reco::BasicClusterCollection::const_iterator it_e=basCluEE->begin(); it_e!=basCluEE->end(); it_e++)	
	{
	  if (it_e->energy()<minEnergy_) continue;
	  double eta=it_e->eta();
	  double phi=it_e->phi();
	  double energy=it_e->energy();
	  double maxNear=-10;
	  double sumNear=0;
	  for (reco::BasicClusterCollection::const_iterator it_in=basCluEB->begin(); it_in!=basCluEB->end(); it_in++)	
	    {    
	      double dPhi;
	      if (it_in->phi()-it_e->phi()>3.14159) dPhi=6.28319-(it_in->phi()-it_e->phi());
	      else dPhi=it_in->phi()-it_e->phi();
	      if (it_in==it_e||(sqrt(pow(it_in->eta()-it_e->eta(),2)+pow(dPhi,2))>coneSize_)) continue;
	      if (maxNear<it_in->energy()) maxNear=it_in->energy();
	      sumNear+=it_in->energy();
	    }
	  for (reco::BasicClusterCollection::const_iterator it_in=basCluEE->begin(); it_in!=basCluEE->end(); it_in++)
	    {
	      double dPhi;
	      if (it_in->phi()-it_e->phi()>3.14159) dPhi=6.28319-(it_in->phi()-it_e->phi());
	      else dPhi=it_in->phi()-it_e->phi();
	      if (it_in==it_e||(sqrt(pow(it_in->eta()-it_e->eta(),2)+pow(dPhi,2))>coneSize_)) continue;
	      if (maxNear<it_in->energy()) maxNear=it_in->energy();
	      sumNear+=it_in->energy();
	    }
	  reco::EcalIsolatedParticleCandidate newca(eta, phi, energy, maxNear, sumNear);
	  eipcCollection->push_back(newca);
	}
    }

  
  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  
  std::auto_ptr<reco::EcalIsolatedParticleCandidateCollection> pOut(eipcCollection);
  iEvent.put(pOut);
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalIsolatedParticleCandidateProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalIsolatedParticleCandidateProducer::endJob() {
}


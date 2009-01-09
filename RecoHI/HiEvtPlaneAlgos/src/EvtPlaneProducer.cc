// -*- C++ -*-
//
// Package:    EvtPlaneProducer
// Class:      EvtPlaneProducer
// 
/**\class EvtPlaneProducer EvtPlaneProducer.cc RecoHI/EvtPlaneProducer/src/EvtPlaneProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sergey Petrushanko
//         Created:  Fri Jul 11 10:05:00 2008
// $Id: EvtPlaneProducer.cc,v 1.1 2008/07/20 19:19:35 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include <iostream>

using namespace std;

//
// class decleration
//

class EvtPlaneProducer : public edm::EDProducer {
   public:
      explicit EvtPlaneProducer(const edm::ParameterSet&);
      ~EvtPlaneProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  bool useECAL_;
  bool useHCAL_;

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
EvtPlaneProducer::EvtPlaneProducer(const edm::ParameterSet& iConfig)
{
   //register your products
  
  useECAL_ = iConfig.getUntrackedParameter<bool>("useECAL",true);
  useHCAL_ = iConfig.getUntrackedParameter<bool>("useHCAL",true);

  produces<reco::EvtPlane>("caloLevel");

}


EvtPlaneProducer::~EvtPlaneProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EvtPlaneProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace HepMC;

      double ugol[9], ugol2[9];
      double tower_eta, tower_phi, tower_energy, tower_energy_e, tower_energy_h;
      double s1t, s2t, s1e, s2e, s1h, s2h;
      double s1[9], s2[9], TEnergy[144], TPhi[144];
      double pi = 3.14159;
      int numb;	

      double planeA     =  0.;

//       cout << endl << "  Start of the event plane determination." << endl;

       for(int j=0;j<9;j++) {
        s1[j]  = 0.;
        s2[j]  = 0.;
       }
      
       for(int l=0;l<144;l++) {
        TEnergy[l]  = 0.;
        TPhi[l]  = 0.;
       }

      Handle<CaloTowerCollection> calotower;
      iEvent.getByLabel("towerMaker",calotower);
      
      if(!calotower.isValid()){
        cout << "Error! Can't get calotower product!" << endl;
       return ;
      }

	for (CaloTowerCollection::const_iterator j = calotower->begin();j !=calotower->end(); j++) {

//        cout << *j << std::endl;
//        cout << "ENERGY HAD " << j->hadEnergy()<< " ENERGY EM " <<j->emEnergy() 
//	  << " ETA " <<j->eta() << " PHI " <<j->phi() << std::endl;
	     
        tower_eta        = j->eta();
        tower_phi        = j->phi();
	tower_energy_e   = j->emEnergy();
	tower_energy_h   = j->hadEnergy();
        tower_energy     = tower_energy_e + tower_energy_h;
	
	s1t = tower_energy*sin(2.*tower_phi-pi);
	s2t = tower_energy*cos(2.*tower_phi-pi);
        s1e = tower_energy_e*sin(2.*tower_phi-pi);
	s2e = tower_energy_e*cos(2.*tower_phi-pi);
	s1h = tower_energy_h*sin(2.*tower_phi-pi);
	s2h = tower_energy_h*cos(2.*tower_phi-pi);

	 if (fabs(tower_eta)<3.){

	  numb = static_cast< int >(72.*(tower_phi/pi + 1.) - 0.5);
	  TEnergy[numb] += tower_energy;
	  TPhi[numb]     = tower_phi;

// barrel + endcap
  	  s1[0] +=  s1t;
	  s2[0] +=  s2t;
  	  s1[3] +=  s1h;
	  s2[3] +=  s2h;
  	  s1[6] +=  s1e;
	  s2[6] +=  s2e;
	  
// endcap
	 if (fabs(tower_eta)>1.5) {
  	  s1[2] +=  s1t;
	  s2[2] +=  s2t;
  	  s1[5] +=  s1h;
	  s2[5] +=  s2h;
  	  s1[8] +=  s1e;
	  s2[8] +=  s2e;
	 }
	 }
	 
// barrel
	 if (fabs(tower_eta)<1.5){
  	  s1[1] +=  s1t;
	  s2[1] +=  s2t;
  	  s1[4] +=  s1h;
	  s2[4] +=  s2h;
  	  s1[7] +=  s1e;
	  s2[7] +=  s2e;
	 }
	}
	
      for(int j1=0;j1<9;j1++) {
 
       if (s2[j1]==0.) {ugol[j1]=0.;}
       else {ugol[j1] = 0.5*atan(s1[j1]/s2[j1]);}
       
       if ( s2[j1] < 0 && s1[j1] <  0) ugol[j1] = ugol[j1] - pi/2.;
       if ( s2[j1] < 0 && s1[j1] >= 0) ugol[j1] = ugol[j1] + pi/2.;
       
       ugol2[j1] = ugol[j1] + pi/2.;
       if (ugol2[j1]>pi/2.) ugol2[j1] = ugol2[j1] - pi;
       
      }

/*       
       cout <<  endl << "   Azimuthal angle of reaction plane (with minimum)" << endl
       << "HCAL+ECAL (b+e)   " << ugol[0] << endl
       << "HCAL+ECAL (b)     " << ugol[1] << endl
       << "HCAL+ECAL (e)     " << ugol[2] << endl
       << "HCAL      (b+e)   " << ugol[3] << endl
       << "HCAL      (b)     " << ugol[4] << endl
       << "HCAL      (e)     " << ugol[5] << endl
       << "ECAL      (b+e)   " << ugol[6] << endl
       << "ECAL      (b)     " << ugol[7] << endl
       << "ECAL      (e)     " << ugol[8] << endl;

       cout <<  endl << "   Azimuthal angle of reaction plane (with maximum)" << endl
       << "HCAL+ECAL (b+e)   " << ugol2[0] << endl
       << "HCAL+ECAL (b)     " << ugol2[1] << endl
       << "HCAL+ECAL (e)     " << ugol2[2] << endl
       << "HCAL      (b+e)   " << ugol2[3] << endl
       << "HCAL      (b)     " << ugol2[4] << endl
       << "HCAL      (e)     " << ugol2[5] << endl
       << "ECAL      (b+e)   " << ugol2[6] << endl
       << "ECAL      (b)     " << ugol2[7] << endl
       << "ECAL      (e)     " << ugol2[8] << endl  << endl;

*/

   if(useECAL_ && !useHCAL_){
     planeA=ugol2[6];
     std::auto_ptr<EvtPlane> evtplaneOutput(new EvtPlane(planeA));
     iEvent.put(evtplaneOutput, "caloLevel");   
   }

   if(useHCAL_ && !useECAL_){
     planeA=ugol2[3];
     std::auto_ptr<EvtPlane> evtplaneOutput(new EvtPlane(planeA));
     iEvent.put(evtplaneOutput, "caloLevel");   
   }   

   if(useECAL_ && useHCAL_){
     planeA=ugol2[0];
     std::auto_ptr<EvtPlane> evtplaneOutput(new EvtPlane(planeA));
     iEvent.put(evtplaneOutput,"caloLevel");   
   }   	

// cout << "  "<< planeA << endl;
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
EvtPlaneProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EvtPlaneProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EvtPlaneProducer);

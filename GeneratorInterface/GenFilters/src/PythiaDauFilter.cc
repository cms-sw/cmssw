
#include "GeneratorInterface/GenFilters/interface/PythiaDauFilter.h"


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/PythiaWrapper6_4.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaDauFilter::PythiaDauFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
chargeconju(iConfig.getUntrackedParameter("ChargeConjugation", true)),
ndaughters(iConfig.getUntrackedParameter("NumberDaughters", 0)),
minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
maxptcut(iConfig.getUntrackedParameter("MaxPt", 14000.)),
minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.))
{
   //now do what ever initialization is needed
   vector<int> defdauID;
   defdauID.push_back(0);
   dauIDs = iConfig.getUntrackedParameter< vector<int> >("DaughterIDs",defdauID);
}


PythiaDauFilter::~PythiaDauFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaDauFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
    
   
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {
     
     if( (*p)->pdg_id() != particleID ) continue ;
     int ndauac = 0;
     int ndau = 0;     
     if ( (*p)->end_vertex() ) {	
       for ( HepMC::GenVertex::particle_iterator 
	       des=(*p)->end_vertex()->particles_begin(HepMC::children);
	     des != (*p)->end_vertex()->particles_end(HepMC::children);
	     ++des ) {
	 ++ndau;       
	 for(int dauID : dauIDs) {
	   if( (*des)->pdg_id() != dauID ) continue ;
	   if(   (*des)->momentum().perp() >  minptcut  &&
		 (*des)->momentum().perp() <  maxptcut  &&
		 (*des)->momentum().eta()  >  minetacut && 
		 (*des)->momentum().eta()  <  maxetacut ) {
	     ++ndauac;
	     break;
	   } 
	 }	       		     
       }
     }  
     if( ndau ==  ndaughters && ndauac == ndaughters ) {
       accepted = true;
       break;
     }    
     
   }
   
   
   if( !accepted && chargeconju ) {
     
     for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	   p != myGenEvent->particles_end(); ++p ) {
       
       if( (*p)->pdg_id() != -particleID ) continue ;
       int ndauac = 0;
       int ndau = 0;     
       if ( (*p)->end_vertex() ) {
	 for ( HepMC::GenVertex::particle_iterator 
		 des=(*p)->end_vertex()->particles_begin(HepMC::children);
	       des != (*p)->end_vertex()->particles_end(HepMC::children);
	       ++des ) {
	   ++ndau;
	   for(int & dauID : dauIDs) {
	     int IDanti = -dauID;
	     int pythiaCode = PYCOMP(dauID);
	     int has_antipart = pydat2.kchg[3-1][pythiaCode-1];
	     if( has_antipart == 0 ) IDanti = dauID;
	     if( (*des)->pdg_id() != IDanti ) continue ;
	     if(   (*des)->momentum().perp() >  minptcut  &&
		   (*des)->momentum().perp() <  maxptcut  &&
		   (*des)->momentum().eta()  >  minetacut && 
		   (*des)->momentum().eta()  <  maxetacut ) {
	       ++ndauac;
	       break;
	     } 
	   }	       		     
	 }
       }
       if( ndau ==  ndaughters && ndauac == ndaughters ) {
	 accepted = true;
	 break;
       }    
     }
     
   }    

   if (accepted){
   return true; } else {return false;}

}

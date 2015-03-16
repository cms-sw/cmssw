
#include "GeneratorInterface/GenFilters/interface/PythiaFilter.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaFilter::PythiaFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")))),
particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
minpcut(iConfig.getUntrackedParameter("MinP", 0.)),
maxpcut(iConfig.getUntrackedParameter("MaxP", 10000.)),
minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)),
minrapcut(iConfig.getUntrackedParameter("MinRapidity", -20.)),
maxrapcut(iConfig.getUntrackedParameter("MaxRapidity", 20.)),
minphicut(iConfig.getUntrackedParameter("MinPhi", -3.5)),
maxphicut(iConfig.getUntrackedParameter("MaxPhi", 3.5)),
status(iConfig.getUntrackedParameter("Status", 0)),
motherID(iConfig.getUntrackedParameter("MotherID", 0)),
processID(iConfig.getUntrackedParameter("ProcessID", 0))
{
   //now do what ever initialization is needed

}


PythiaFilter::~PythiaFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
    
   if(processID == 0 || processID == myGenEvent->signal_process_id()) {
     
     for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	   p != myGenEvent->particles_end(); ++p ) {
       
       rapidity = 0.5*log( ((*p)->momentum().e()+(*p)->momentum().pz()) / ((*p)->momentum().e()-(*p)->momentum().pz()) );
       
       if ( abs((*p)->pdg_id()) == particleID 
	    && (*p)->momentum().rho() > minpcut 
	    && (*p)->momentum().rho() < maxpcut
	    && (*p)->momentum().perp() > minptcut 
	    && (*p)->momentum().perp() < maxptcut
	    && (*p)->momentum().eta() > minetacut
	    && (*p)->momentum().eta() < maxetacut 
	    && rapidity > minrapcut
	    && rapidity < maxrapcut 
	    && (*p)->momentum().phi() > minphicut
	    && (*p)->momentum().phi() < maxphicut ) {
	 
         
	 
	 if (status == 0 && motherID == 0){
	   accepted = true;
	 }
	 if (status != 0 && motherID == 0){
	   if ((*p)->status() == status)   
	     accepted = true;
	 }
	 
	 HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));
	 
	 if (status == 0 && motherID != 0){    
	   if (abs(mother->pdg_id()) == abs(motherID)) {
	     accepted = true;
	   }
	 }
	 if (status != 0 && motherID != 0){
	   
	   if ((*p)->status() == status && abs(mother->pdg_id()) == abs(motherID)){   
	     accepted = true;
	     
	   }
	 }
	 
	 /*
	   if (status == 0 && motherID != 0){    
	   if (abs(((*p)->mother())->pdg_id()) == abs(motherID)) {
	   accepted = true;
	   }
	   }
	   if (status != 0 && motherID != 0){
           
	   if ((*p)->status() == status && abs(((*p)->mother())->pdg_id()) == abs(motherID)){   
	   accepted = true;
	   
	   }
	   }
	 */
	 
       }    
     }
     
   } else { accepted = true; }

   if (accepted){
   return true; } else {return false;}

}



#include "IOMC/GeneratorInterface/interface/PythiaFilter.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaFilter::PythiaFilter(const edm::ParameterSet& iConfig) :
label_(iConfig.getUntrackedParameter("moduleLabel",std::string("source"))),
particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)),
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
   iEvent.getByLabel(label_, evt);

    HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
    
    if(processID == 0 || processID == myGenEvent->signal_process_id()) {
    
    for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	  p != myGenEvent->particles_end(); ++p ) {
	  
 
	if ( abs((*p)->pdg_id()) == particleID 
	     && (*p)->momentum().perp() > minptcut 
	     && (*p)->momentum().perp() < maxptcut
	     && (*p)->momentum().eta() > minetacut
	     && (*p)->momentum().eta() < maxetacut 
	     && (*p)->momentum().phi() > minphicut
	     && (*p)->momentum().phi() < maxphicut ) {

             
	     
	     if (status == 0 && motherID == 0){
           	     accepted = true;
	     }
	     if (status != 0 && motherID == 0){
           	if ((*p)->status() == status)   
		     accepted = true;
	        }
	     
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
	     
	 }    
	     
	    
	  
    }

    } else { accepted = true; }


    delete myGenEvent; 


   if (accepted){
   return true; } else {return false;}

}


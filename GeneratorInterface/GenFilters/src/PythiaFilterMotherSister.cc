
#include "GeneratorInterface/GenFilters/interface/PythiaFilterMotherSister.h"
#include "GeneratorInterface/GenFilters/interface/MCFilterZboostHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaFilterMotherSister::PythiaFilterMotherSister(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
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
sisterID(iConfig.getUntrackedParameter("SisterID", 0)),
processID(iConfig.getUntrackedParameter("ProcessID", 0)),
betaBoost(iConfig.getUntrackedParameter("BetaBoost",0.))
{
   //now do what ever initialization is needed

}


PythiaFilterMotherSister::~PythiaFilterMotherSister()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaFilterMotherSister::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
    
   if(processID == 0 || processID == myGenEvent->signal_process_id()) {
     
     for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	   p != myGenEvent->particles_end(); ++p ) {
       HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(),betaBoost);
       double rapidity = 0.5*log( (mom.e()+mom.pz()) / (mom.e()-mom.pz()) );
       
       if ( abs((*p)->pdg_id()) == particleID 
	    && mom.rho() > minpcut 
	    && mom.rho() < maxpcut
	    && (*p)->momentum().perp() > minptcut 
	    && (*p)->momentum().perp() < maxptcut
	    && mom.eta() > minetacut
	    && mom.eta() < maxetacut 
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
	
         if(accepted and motherID!=0){ 
           uint good_sis = 0;
           for ( HepMC::GenVertex::particle_iterator dau = mother->end_vertex()->particles_begin(HepMC::children); 
                                                     dau != mother->end_vertex()->particles_end(HepMC::children);
                                                     ++dau ) {
             if(abs((*dau)->pdg_id()) == abs(sisterID)) {
               ++good_sis;
               std::cout << __LINE__ << "]\t found good sister!" << std::endl;
             }
           }
           
           if(good_sis==0) accepted = false;

         } // if mother was (requested and) found
       } // if gen particle has the right id 
     } // loop over gen particles
     
   } else { accepted = true; }

   if (accepted){
   return true; } else {return false;}

}


#include "GeneratorInterface/GenFilters/interface/PythiaFilterMotherGrandMother.h"
#include "GeneratorInterface/GenFilters/interface/MCFilterZboostHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


PythiaFilterMotherGrandMother::PythiaFilterMotherGrandMother(const edm::ParameterSet& iConfig) :
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
//status(iConfig.getUntrackedParameter("Status", 0)),
grandMotherIDs(iConfig.getUntrackedParameter("GrandMotherIDs", std::vector<int>{0})),
motherID(iConfig.getUntrackedParameter("MotherID", 0)),
//processID(iConfig.getUntrackedParameter("ProcessID", 0)),
betaBoost(iConfig.getUntrackedParameter("BetaBoost",0.))
{
   //now do what ever initialization is needed

}


PythiaFilterMotherGrandMother::~PythiaFilterMotherGrandMother()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaFilterMotherGrandMother::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
       
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
         p != myGenEvent->particles_end(); ++p ) {
     HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(),betaBoost);
     double rapidity = 0.5*log( (mom.e()+mom.pz()) / (mom.e()-mom.pz()) );

     //==> slows down instead of speeding up... if (accepted) break; 

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
          && (*p)->momentum().phi() < maxphicut ) 
     {
       
        
       //std::cout << "found muon with right pt/eta cuts,  pt=" << mom.rho() << " status=" << (*p)->status() << std::endl; 
       
       HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));
      
       if(abs(mother->pdg_id()) == abs(motherID)){
         //std::cout << "found good mother" << std::endl;  
         // find grandmother
         HepMC::GenParticle* grandMother = (*(mother->production_vertex()->particles_in_const_begin()));
         //std::cout << "grandmother id " << grandMother->pdg_id() << std::endl;
         for (auto grandMotherID : grandMotherIDs){
           if(abs(grandMother->pdg_id()) == abs(grandMotherID)){
             //std::cout << "found good grandmother" << std::endl;
             accepted = true;
           } // if grand mother was found
         } // loop over grand mother ids
       } // if mother was found 
     } // if gen particle is as requested 
   } // loop over gen particles
   
   //std::cout << "For this event: accepted=" << accepted << "\n" << std::endl;

   if (accepted) {return true;}
   else {return false;}

}

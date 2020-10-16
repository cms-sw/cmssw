
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
//status(iConfig.getUntrackedParameter("Status", 0)),
//motherID(iConfig.getUntrackedParameter("MotherID", 0)),
motherIDs(iConfig.getUntrackedParameter("MotherIDs", std::vector<int>{0})),
sisterID(iConfig.getUntrackedParameter("SisterID", 0)),
//processID(iConfig.getUntrackedParameter("ProcessID", 0)),
betaBoost(iConfig.getUntrackedParameter("BetaBoost",0.)),
maxSisDisplacement(iConfig.getUntrackedParameter("MaxSisterDisplacement", -1.))
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
      
       // check various possible mothers
       for(auto motherID : motherIDs){

         if(abs(mother->pdg_id()) == abs(motherID)){
  
           //HepMC::FourVector mom_mum = MCFilterZboostHelper::zboost(mother->momentum(),betaBoost);
           //std::cout << "  found good mother (B meson)! pt=" <<  mom_mum.rho() << std::endl;
  
           // loop over its daughters
           for ( HepMC::GenVertex::particle_iterator dau = mother->end_vertex()->particles_begin(HepMC::children); 
                                                     dau != mother->end_vertex()->particles_end(HepMC::children);
                                                     ++dau ) {
             //==> slows down instead of speeding up... if(accepted) break;
             //HepMC::FourVector mom_dau = MCFilterZboostHelper::zboost((*dau)->momentum(),betaBoost);
             //std::cout << "    daughter of B meson " << (*dau)->pdg_id() << " pt=" << mom_dau.rho() << " status=" << (*dau)->status() << std::endl;
             // find the daugther you're interested in
             if(abs((*dau)->pdg_id()) == abs(sisterID)) {
               //std::cout << "      found good sister!" << std::endl;
               /*for(HepMC::GenVertex::particle_iterator dau_hnl = (*dau)->end_vertex()->particles_begin(HepMC::children);
                   dau_hnl != (*dau)->end_vertex()->particles_end(HepMC::children);
                   ++dau_hnl) {
                   HepMC::FourVector mom_dau_hnl = MCFilterZboostHelper::zboost((*dau_hnl)->momentum(),betaBoost);
                   std::cout << "        daughter of HNL " << (*dau_hnl)->pdg_id() << " pt=" << mom_dau_hnl.rho() << " status=" << (*dau_hnl)->status() << std::endl;    
               }*/
               
               // calculate displacement wrt B, need the orig vertex of the trigger muon 
               //HepMC::GenVertex* v1_B = mother->end_vertex(); // where the B decays
               HepMC::GenVertex* v1   = (*dau)->production_vertex();
               HepMC::GenVertex* v2   = (*dau)->end_vertex();

               double lx12 = v1->position().x() - v2->position().x();
               double ly12 = v1->position().y() - v2->position().y();
               double lz12 = v1->position().z() - v2->position().z();
               //double lxy12 =  sqrt( lx12*lx12 + ly12*ly12);
               double lxyz12 = sqrt( lx12*lx12 + ly12*ly12 + lz12*lz12 );
               //std::cout << "Lxyz from HNL vertices: " << lxyz12 << "   " << std::endl;
               //std::cout << "Unit of length: " << HepMC::Units::name(myGenEvent->length_unit()) << std::endl ;
               if(maxSisDisplacement!= -1){
                 if(lxyz12 < maxSisDisplacement){
                   accepted = true;
                 }
               } else {
                   accepted = true;
               }

             } // if sister was found
           } // loop over daughters
         } // if mother was found 
       } // loop over possible mother ids
     } // if gen particle is as requested 
   } // loop over gen particles
   
   //std::cout << "For this event: accepted=" << accepted << "\n" << std::endl;

   if (accepted) {return true;}
   else {return false;}

}

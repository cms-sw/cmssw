
#include <iostream>

// #include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "GeneratorInterface/ExternalDecays/interface/PhotosInterface.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
// #include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"

using namespace gen;
using namespace edm;
using namespace std;


extern "C"{

   void phoini_( void );
   void photos_( int& );

   double phoran_(int *idummy)
   {
      return decayRandomEngine->flat();
   }
/*
   double phoranc_(int *idummy)
   {
      return decayRandomEngine->flat();
   }
*/

   extern struct {
      // bool qedrad[NMXHEP];
      bool qedrad[4000]; // hardcoded for now...
   } phoqed_;

}


PhotosInterface::PhotosInterface()
   : fOnlyPDG(-1)
{
   fSpecialSettings.push_back("QED-brem-off:all");
   fAvoidTauLeptonicDecays = false;
   fIsInitialized = false; 
}

PhotosInterface::PhotosInterface( const edm::ParameterSet& )
   : fOnlyPDG(-1)
{
   fSpecialSettings.push_back("QED-brem-off:all");
   fIsInitialized = false;
}

/*  -->
void PhotosInterface::configureOnlyFor( int ipdg )
{

   fOnlyPDG = ipdg;
   std::ostringstream command;
   command << "QED-brem-off:" << fOnlyPDG ;
   fSpecialSettings.clear();
   fSpecialSettings.push_back( command.str() );
   
   return;

}
*/

void PhotosInterface::init()
{
   
   if ( fIsInitialized ) return; // do init only once
   
   phoini_();
   
   fIsInitialized = true; 

   return;
}

HepMC::GenEvent* PhotosInterface::apply( HepMC::GenEvent* evt )
{
   
   
   if ( !fIsInitialized ) return evt; // conv.read_next_event();
      
   // loop over HepMC::GenEvent, find vertices
      
   for ( int ip=0; ip<evt->particles_size(); ip++ )
   {
      phoqed_.qedrad[ip]=true;
   }
   
   
   // variables for special treatment of tau leptonic decays
   //
   bool tau_leptonic_decay = false;
   int iTauDescCounter = 0;
   int nTauDesc = 0;
   
   for ( int iv=1; iv<=evt->vertices_size(); iv++ )
   {
      
      bool legalVtx = false;
      
      HepMC::GenVertex* vtx = evt->barcode_to_vertex( -iv ) ;
      
      if ( vtx->particles_in_size() != 1 ) continue; // more complex than we need
      if ( vtx->particles_out_size() <= 1 ) continue; // no outcoming particles
      
      if ( (*(vtx->particles_in_const_begin()))->pdg_id() == 111 ) continue; // pi0 decay vtx - no point to try
      
      // --> if ( fOnlyPDG !=-1 && (*(vtx->particles_in_const_begin()))->pdg_id() == fOnlyPDG ) continue;
      
      for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
            pitr != vtx->particles_end(HepMC::children); ++pitr) 
      {

	 // quark or gluon out of this vertex - no good !
	 if ( abs((*pitr)->pdg_id()) >=1 &&  abs((*pitr)->pdg_id()) <=8 ) break;
	 if ( abs((*pitr)->pdg_id()) == 21 ) break;

         if ( (*pitr)->status() == 1 || (*pitr)->end_vertex() )
	 {
	    // OK, legal already !
	    legalVtx = true;
	    break;
	 }
      }
      
      if ( !legalVtx ) continue;
      
      // now do all the loops again
      //
      // first, flush out HEPEVT & tmp barcode storage
      //
      HepMC::HEPEVT_Wrapper::zero_everything();
      fBarcodes.clear();
      
      // add incoming particle
      //
      int index = 1;      
      HepMC::HEPEVT_Wrapper::set_id( index, (*(vtx->particles_in_const_begin()))->pdg_id() );
      HepMC::FourVector vec4;
      vec4 = (*(vtx->particles_in_const_begin()))->momentum();
      HepMC::HEPEVT_Wrapper::set_momentum( index, vec4.x(), vec4.y(), vec4.z(), vec4.e() );
      HepMC::HEPEVT_Wrapper::set_mass( index, (*(vtx->particles_in_const_begin()))->generated_mass() );
      HepMC::HEPEVT_Wrapper::set_position( index, vtx->position().x(), vtx->position().y(),
                                                  vtx->position().z(), vtx->position().t() );
      HepMC::HEPEVT_Wrapper::set_status( index, (*(vtx->particles_in_const_begin()))->status() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 );
      fBarcodes.push_back( (*(vtx->particles_in_const_begin()))->barcode() );
                  
      // special case: avoid tau leptonic decays
      //

      if ( fAvoidTauLeptonicDecays && !tau_leptonic_decay && abs((*(vtx->particles_in_const_begin()))->pdg_id()) == 15 )
      {      
         for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
               pitr != vtx->particles_end(HepMC::children); ++pitr) 
         {
	    if ( abs((*pitr)->pdg_id()) == 11 || abs((*pitr)->pdg_id()) == 13 ) // leptonic decay !!!
	                                                                        // do brem off tau but NOT off decay products
	    {
	       tau_leptonic_decay = true;
	       break;
	    }	 
         }
         if ( vtx->particles_begin(HepMC::children) == vtx->particles_begin(HepMC::descendants) && 
              vtx->particles_end(HepMC::children) == vtx->particles_end(HepMC::descendants) ) // FIXME !!!!!
	                                                                                      // Maybe better vtx nested loop(s) 
											      // instead of "descendants" ???
         {
	    nTauDesc = vtx->particles_out_size();
         }
	 else
	 {
            for ( HepMC::GenVertex::particle_iterator pitr1=vtx->particles_begin(HepMC::children);
                  pitr1 != vtx->particles_end(HepMC::children); ++pitr1) 
            {
	       nTauDesc++;
	    }
	 }
         // this is just the 1st tau in the branch, so it's allowed to emit
	 phoqed_.qedrad[index-1]=true;
	 iTauDescCounter = 0;
      }
           
      // add outcoming particles (decay products)
      //
      int lastDau = 1;
      for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
            pitr != vtx->particles_end(HepMC::children); ++pitr) 
      {

	 if ( (*pitr)->status() == 1 || (*pitr)->end_vertex() )
	 {
	    index++;
	    vec4 = (*pitr)->momentum();
	    HepMC::HEPEVT_Wrapper::set_id( index, (*pitr)->pdg_id() );
            HepMC::HEPEVT_Wrapper::set_momentum( index, vec4.x(), vec4.y(), vec4.z(), vec4.e() );
	    HepMC::HEPEVT_Wrapper::set_mass( index, (*pitr)->generated_mass() );
	    vec4 = (*pitr)->production_vertex()->position();
            HepMC::HEPEVT_Wrapper::set_position( index, vec4.x(), vec4.y(), vec4.z(), vec4.t() );
	    HepMC::HEPEVT_Wrapper::set_status( index, (*pitr)->status() );
	    HepMC::HEPEVT_Wrapper::set_parents( index, 1, 1 );
	    fBarcodes.push_back( (*pitr)->barcode() );
	    lastDau++;
	    if ( fAvoidTauLeptonicDecays && tau_leptonic_decay )
	    {
	       phoqed_.qedrad[index-1]=false;
	       iTauDescCounter++;
	    }
	 }
      }
            
      // store, further to set NHEP in HEPEVT
      //
      int nentries = index;
      
      // reset master pointer to mother
      index = 1;
      HepMC::HEPEVT_Wrapper::set_children ( index, 2, lastDau ); // FIXME: need to check 
                                                                 // if last daughter>=2 !!!
      
      // finally, set number of entries (NHEP) in HEPEVT
      //
      HepMC::HEPEVT_Wrapper::set_number_entries( nentries );

      // cross-check printout HEPEVT
      // HepMC::HEPEVT_Wrapper::print_hepevt();
     
      // OK, 1-level vertex is formed - now, call PHOTOS
      //
      photos_( index ) ;
      
      // another cross-check printout HEPEVT - after photos
      // HepMC::HEPEVT_Wrapper::print_hepevt();


      // now check if something has been generated 
      // and make all adjustments to underlying vtx/parts
      //
      attachParticles( evt, vtx, nentries );

      // ugh, done with this vertex !
      
      // now some resets
      //
      if ( fAvoidTauLeptonicDecays && tau_leptonic_decay && iTauDescCounter == nTauDesc ) // treated tau leptonic decay and have come to the last descendent
      {
         tau_leptonic_decay = false;
      }
   
   }
   
   // restore event number in HEPEVT (safety measure, somehow needed by Hw6)
   HepMC::HEPEVT_Wrapper::set_event_number( evt->event_number() );

   return evt;
      
}

void PhotosInterface::attachParticles( HepMC::GenEvent* evt, HepMC::GenVertex* vtx, int nentries )
{

   if ( HepMC::HEPEVT_Wrapper::number_entries() > nentries ) 
   {
         // yes, need all corrections and adjustments -
	 // figure out how many photons and what particles in 
	 // the decay branch have changes;
	 // also, follow up each one and correct accordingly;
	 // at the same time, add photon(s) to the GenVertex
	 //	 
	 
	 // vtx->print();
	 
	 int largestBarcode = -1;
	 int Nbcodes = fBarcodes.size();
	 
	 for ( int ip=1; ip<Nbcodes; ip++ )
	 {

	    int bcode = fBarcodes[ip];
	    HepMC::GenParticle* prt = evt->barcode_to_particle( bcode );
	    if ( bcode > largestBarcode ) largestBarcode = bcode;
	    double px = HepMC::HEPEVT_Wrapper::px(ip+1);
	    double py = HepMC::HEPEVT_Wrapper::py(ip+1);
	    double pz = HepMC::HEPEVT_Wrapper::pz(ip+1);
	    double e  = HepMC::HEPEVT_Wrapper::e(ip+1);
	    double m  = HepMC::HEPEVT_Wrapper::m(ip+1);	  
	    	    
	    if ( prt->end_vertex() )
	    {
	       
	       HepMC::GenVertex* endVtx = prt->end_vertex();
	       
               std::vector<int> secVtxStorage;
               secVtxStorage.clear();
	       
	       secVtxStorage.push_back( endVtx->barcode() );
	    
	       HepMC::FourVector mom4 = prt->momentum();
	    
	       // now rescale all descendants
	       double bet1[3], bet2[3], gam1, gam2, pb;
	       double mass = mom4.m();
	       bet1[0] = -(mom4.px()/mass);
	       bet1[1] = -(mom4.py()/mass);
	       bet1[2] = -(mom4.pz()/mass);
	       bet2[0] = px/m;
	       bet2[1] = py/m;
	       bet2[2] = pz/m;
	       gam1 = mom4.e()/mass;
	       gam2 = e/m;
	    
	       unsigned int vcounter = 0;
	       	    
               while ( vcounter < secVtxStorage.size() )
	       {
	          
		  HepMC::GenVertex* theVtx = evt->barcode_to_vertex( secVtxStorage[vcounter] );
		  	          
		  for ( HepMC::GenVertex::particle_iterator pitr=theVtx->particles_begin(HepMC::children);
                        pitr != theVtx->particles_end(HepMC::children); ++pitr) 
                  {
	       
		     if ( (*pitr)->end_vertex() )
		     {
		        secVtxStorage.push_back( (*pitr)->end_vertex()->barcode() );
		     }
		     
		     if ( theVtx->particles_out_size() == 1 && (*pitr)->pdg_id() == prt->pdg_id() ) 
		     {
		        // carbon copy
			(*pitr)->set_momentum( HepMC::FourVector(px,py,pz,e) );
			continue;
		     }
		     	       
	             HepMC::FourVector dmom4 = (*pitr)->momentum();
	       
	             // Boost vector to parent rest frame...
	             pb = bet1[0]*dmom4.px() + bet1[1]*dmom4.py() + bet1[2]*dmom4.pz();
	             double dpx = dmom4.px() + bet1[0] * (dmom4.e() + pb/(gam1+1.) );
	             double dpy = dmom4.py() + bet1[1] * (dmom4.e() + pb/(gam1+1.) );
	             double dpz = dmom4.pz() + bet1[2] * (dmom4.e() + pb/(gam1+1.) );
	             double de  = gam1*dmom4.e() + pb;
	             // ...and boost back to modified parent frame
	             pb = bet2[0]*dpx + bet2[1]*dpy + bet2[2]*dpz;
	             dpx += bet2[0] * ( de + pb/(gam2+1.) );
	             dpy += bet2[1] * ( de + pb/(gam2+1.) );
	             dpz += bet2[2] * ( de + pb/(gam2+1.) );
	             de *= gam2;
	             de += pb;
	       
	             (*pitr)->set_momentum( HepMC::FourVector(dpx,dpy,dpz,de) );
		     
	          }		  		  
		  vcounter++;	    
               }
	       
	       secVtxStorage.clear();
	    }
	    
	    prt->set_momentum( HepMC::FourVector(px,py,pz,e) );
	    
	 } // ok, all affected particles update, but the photon(s) still not inserted
	 	 
	 int newlyGen =  HepMC::HEPEVT_Wrapper::number_entries() - nentries;
	 
	 if ( largestBarcode < evt->particles_size() )
	 {
	    // need to adjust barcodes down from the affected vertex/particles
	    // such that we "free up" barcodes for newly generated photons
	    // in the middle of the event record
	    //
	    for ( int ipp=evt->particles_size(); ipp>largestBarcode; ipp-- )
	    {
	       (evt->barcode_to_particle(ipp))->suggest_barcode( ipp+newlyGen );
	    }
	 }
	 
	 // now attach new generated photons to the vertex
	 //
	 for ( int ipnw=1; ipnw<=newlyGen; ipnw++ )
	 {
	     int nbcode = largestBarcode+ipnw;
	     int pdg_id = HepMC::HEPEVT_Wrapper::id( nentries+ipnw );
	     int status = HepMC::HEPEVT_Wrapper::status( nentries+ipnw );
	     double px  =  HepMC::HEPEVT_Wrapper::px( nentries+ipnw );
	     double py  =  HepMC::HEPEVT_Wrapper::py( nentries+ipnw );
	     double pz  =  HepMC::HEPEVT_Wrapper::pz( nentries+ipnw );
	     double e   =  HepMC::HEPEVT_Wrapper::e(  nentries+ipnw );
	     double m   =  HepMC::HEPEVT_Wrapper::m(  nentries+ipnw );
	  
	     HepMC::GenParticle* NewPart = new HepMC::GenParticle( HepMC::FourVector(px,py,pz,e),
	                                                            pdg_id, status);
	     NewPart->set_generated_mass( m );
	     NewPart->suggest_barcode( nbcode );
	     vtx->add_particle_out( NewPart ) ;
	 }  
   
      // vtx->print();
      // std::cout << " leaving attachParticles() " << std::endl;
   
   } // end of global if-statement 

   return;
}

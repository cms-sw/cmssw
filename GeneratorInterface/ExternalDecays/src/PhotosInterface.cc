
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

void PhotosInterface::configureOnlyFor( int ipdg )
{

   fOnlyPDG = ipdg;
//   std::ostringstream command;
//   command << "QED-brem-off:" << fOnlyPDG ;
   fSpecialSettings.clear();
//   fSpecialSettings.push_back( command.str() );
   
   return;

}

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
      
   // for ( int ip=0; ip<evt->particles_size(); ip++ )
   for ( int ip=0; ip<4000; ip++ ) // 4000 is the max size of the array
   {
      phoqed_.qedrad[ip]=true;
   }
      
   //
   // now do actual job
   //
   
   for ( int iv=1; iv<=evt->vertices_size(); iv++ )
   {
      
      bool legalVtx = false;
      
      fSecVtxStore.clear();
      
      HepMC::GenVertex* vtx = evt->barcode_to_vertex( -iv ) ;
      
      if ( vtx->particles_in_size() != 1 ) continue; // more complex than we need
      if ( vtx->particles_out_size() <= 1 ) continue; // no outcoming particles
      
      if ( (*(vtx->particles_in_const_begin()))->pdg_id() == 111 ) continue; // pi0 decay vtx - no point to try
      
      if ( fOnlyPDG != 1 && (*(vtx->particles_in_const_begin()))->pdg_id() != fOnlyPDG )
      {
         continue;
      }
      else
      {
         // requested for specific PDG ID only, typically tau (15)
	 //
	 // first check if a brem vertex, where outcoming are the same pdg id and a photon
	 //
	 bool same = false;
	 for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
               pitr != vtx->particles_end(HepMC::children); ++pitr)
         {
	    if ( (*pitr)->pdg_id() == fOnlyPDG )
	    {
	       same = true;
	       break;
	    }
	 }
	 if ( same ) continue;
	 
	 // OK, we get here if incoming fOnlyPDG and something else outcoming
	 // call it for the whole branch starting at vtx barcode iv, and go on
	 // NOTE: theoretically, it has a danger of double counting in vertices
	 // down the decay branch originating from fOnlyPDG, but in practice
	 // it's unlikely that down the branchg there'll be more fOnlyPDG's
	 
	 // cross-check printout
	 // vtx->print();
	 
	 // overprotection...
	 //
	 if ( fOnlyPDG == 15 && fAvoidTauLeptonicDecays && isTauLeptonicDecay( vtx ) ) continue; 
	 
	 applyToBranch( evt, -iv );
	 continue;
      }
      
      // configured for all types of particles
      //
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
      
      applyToVertex( evt, -iv );

   } // end of master loop
   
   // restore event number in HEPEVT (safety measure, somehow needed by Hw6)
   HepMC::HEPEVT_Wrapper::set_event_number( evt->event_number() );

   return evt;
      
}

void PhotosInterface::applyToVertex( HepMC::GenEvent* evt, int vtxbcode )
{

   HepMC::GenVertex* vtx = evt->barcode_to_vertex( vtxbcode );
   
   if ( fAvoidTauLeptonicDecays && isTauLeptonicDecay( vtx ) ) return;

   // cross-check printout
   //
   // vtx->print();
            
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
	 }
         if ( (*pitr)->end_vertex() )
         {
            fSecVtxStore.push_back( (*pitr)->end_vertex()->barcode() );
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
   //  HepMC::HEPEVT_Wrapper::print_hepevt();
     
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

   return;

}

void PhotosInterface::applyToBranch( HepMC::GenEvent* evt, int vtxbcode )
{
  
   
   fSecVtxStore.clear();
      
   // 1st level vertex
   //
   applyToVertex( evt, vtxbcode );
   
   // now look down the branch for more vertices, if any  
   //
   // Note: fSecVtxStore gets filled up in applyToVertex, if necessary 
   //
   unsigned int vcounter = 0;  
   
   while ( vcounter < fSecVtxStore.size() )
   { 
      applyToVertex( evt, fSecVtxStore[vcounter] );
      vcounter++;
   }
   
   fSecVtxStore.clear(); 

   return;

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
   
      //vtx->print();
      //std::cout << " leaving attachParticles() " << std::endl;
   
   } // end of global if-statement 

   return;
}

bool PhotosInterface::isTauLeptonicDecay( HepMC::GenVertex* vtx )
{

   if ( abs((*(vtx->particles_in_const_begin()))->pdg_id()) != 15 ) return false;
   
   for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
         pitr != vtx->particles_end(HepMC::children); ++pitr) 
   {
      if ( abs((*pitr)->pdg_id()) == 11 || abs((*pitr)->pdg_id()) == 13 )
      {
         return true;
      }      
   } 
   
   return false;  

}

/* very first version... but Phptos seems to want a SINGLE vertex, nor a branch

void PhotosInterface::applyToBranch( HepMC::GenEvent* evt, int vtxbcode )
{

   HepMC::GenVertex* vtx = evt->barcode_to_vertex( vtxbcode );
   
   vtx->print();
   
   // special case - do nothing
   // we don't brem off tau since it'll be done by master generator,
   // and within tau leptonic decays the brem is done by tauola
   //
   if ( fAvoidTauLeptonicDecays && isTauLeptonicDecay( vtx ) ) return;
   
   std::vector<int> secVtxStore;
   secVtxStore.clear();
   
   // first, flush out HEPEVT & tmp barcode storage
   //
   HepMC::HEPEVT_Wrapper::zero_everything();
   fBarcodes.clear();

   // form 1st level vertex
   //
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

   int lastDau = 1;
   for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
         pitr != vtx->particles_end(HepMC::children); ++pitr) 
   {

      // put particles into HEPEVT - form 1st level vertex
      //
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
      }
      
      if ( (*pitr)->end_vertex() )
      {
         secVtxStore.push_back( (*pitr)->end_vertex()->barcode() );
      }      
   }
      
   if ( lastDau < 2 ) lastDau = 2;
   HepMC::HEPEVT_Wrapper::set_children ( 1, 2, lastDau );                                                           
      
   // now look down the branch for more vertices, if any
   //
   unsigned int vcounter = 0;  
   int firstDau = lastDau + 1;  
   int index1 = index;   	    
   while ( vcounter < secVtxStore.size() )
   { 
      HepMC::GenVertex* theVtx = evt->barcode_to_vertex( secVtxStore[vcounter] );
      for ( HepMC::GenVertex::particle_iterator pitr1=theVtx->particles_begin(HepMC::children);
                        pitr1 != theVtx->particles_end(HepMC::children); ++pitr1) 
      {
         if ( (*pitr1)->status() == 1 || (*pitr1)->end_vertex() )
         {
	    index++;
	    vec4 = (*pitr1)->momentum();
	    HepMC::HEPEVT_Wrapper::set_id( index, (*pitr1)->pdg_id() );
            HepMC::HEPEVT_Wrapper::set_momentum( index, vec4.x(), vec4.y(), vec4.z(), vec4.e() );
	    HepMC::HEPEVT_Wrapper::set_mass( index, (*pitr1)->generated_mass() );
	    vec4 = (*pitr1)->production_vertex()->position();
            HepMC::HEPEVT_Wrapper::set_position( index, vec4.x(), vec4.y(), vec4.z(), vec4.t() );
	    HepMC::HEPEVT_Wrapper::set_status( index, (*pitr1)->status() );
	    HepMC::HEPEVT_Wrapper::set_parents( index, index1, index1 );
	    fBarcodes.push_back( (*pitr1)->barcode() );
	    lastDau++;
         }
         if ( (*pitr1)->end_vertex() )
         {
            secVtxStore.push_back( (*pitr1)->end_vertex()->barcode() );
	 }
      }
      index1 += 1;
      HepMC::HEPEVT_Wrapper::set_children ( index1, firstDau, lastDau );
      index1 = index;
      firstDau = lastDau + 1;
      vcounter++;
   } 
   
   // finally, set number of entries (NHEP) in HEPEVT
   //
   int nentries = index;
   HepMC::HEPEVT_Wrapper::set_number_entries( nentries );  

   // test printout
   HepMC::HEPEVT_Wrapper::print_hepevt();   

   // don't start from the first one since it's going to be
   // the incoming particles (e.g. tau) which is already
   // treated by the brem from master generator
   //
   index = 2;
   // oh well, maybe 2 was a bad idea...
   index = 1;
   photos_( index );

   HepMC::HEPEVT_Wrapper::print_hepevt();   
   
   attachParticles( evt, vtx, nentries );
     
   return;

}
*/


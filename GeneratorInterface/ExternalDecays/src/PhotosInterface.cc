
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

}


PhotosInterface::PhotosInterface()
   : fOnlyPDG(-1)
{
   fSpecialSettings.push_back("QED-brem-off:all");
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
   
   // event record convertor 
   // ...well, I'm not sure we need it here, 
   // as we do it by hands, only part of the record
   //
   // HepMC::IO_HEPEVT conv;
   
   if ( !fIsInitialized ) return evt; // conv.read_next_event();
   
   // cross-check printout HepMC::GenEvent
   //
   //evt->print();
   
   // int numPartBefore = HepMC::HEPEVT_Wrapper::number_entries();
   // HepMC::HEPEVT_Wrapper::print_hepevt();
   
   // loop over HepMC::GenEvent, find vertices

   std::vector<int> barcodes;
   
   //std::vector<double> scaleFactors;
   // std::vector<HepMC::FourVector> scaleFactors;
   std::vector<Scaling> scaleFactors;
   for ( int ip=0; ip<evt->particles_size(); ip++ )
   {
      scaleFactors.push_back( Scaling(HepMC::ThreeVector(1.,1.,1.),1) );
   }
   
   for ( int iv=1; iv<=evt->vertices_size(); iv++ )
   {
      
      HepMC::GenVertex* vtx = evt->barcode_to_vertex( -iv ) ;
      if ( vtx->particles_in_size() != 1 ) continue; // more complex than we need
      if ( vtx->particles_out_size() <= 0 ) continue; // no outcoming particles
      // --> if ( fOnlyPDG !=-1 && (*(vtx->particles_in_const_begin()))->pdg_id() == fOnlyPDG ) continue;
      // now find at least one "legal" daughter
      bool legalVtx = false;
      for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
            pitr != vtx->particles_end(HepMC::children); ++pitr) 
      {
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
      barcodes.clear();
      
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
      barcodes.push_back( (*(vtx->particles_in_const_begin()))->barcode() );
      
      // check if mother has ever been "altered" !
      //
      int mbcode =  (*(vtx->particles_in_const_begin()))->barcode();
      
      // add outcoming particles (decay products)
      //
      int lastDau = 1;
      for ( HepMC::GenVertex::particle_iterator pitr=vtx->particles_begin(HepMC::children);
            pitr != vtx->particles_end(HepMC::children); ++pitr) 
      {
         if ( scaleFactors[mbcode-1].flag != 1. )
	 {
	    // yes, mother has been changed - adjust daughters
	    
	    vec4 = (*pitr)->momentum();
	    double mass2 = vec4.m2();
	    double pxn = vec4.px() * scaleFactors[mbcode-1].weights.x();
	    double pyn = vec4.py() * scaleFactors[mbcode-1].weights.y();
	    double pzn = vec4.pz() * scaleFactors[mbcode-1].weights.z();
	    double en  = sqrt( pxn*pxn + pyn*pyn + pzn*pzn + mass2 );
	    (*pitr)->set_momentum( HepMC::FourVector(pxn,pyn,pzn,en) );
	    int curbcode = (*pitr)->barcode();
	    double scale = scaleFactors[curbcode-1].weights.x();
	    scaleFactors[curbcode-1].weights.setX( scale*scaleFactors[mbcode-1].weights.x() );
	    scale = scaleFactors[curbcode-1].weights.y();
	    scaleFactors[curbcode-1].weights.setY( scale*scaleFactors[mbcode-1].weights.y() );
	    scale = scaleFactors[curbcode-1].weights.z();
	    scaleFactors[curbcode-1].weights.setZ( scale*scaleFactors[mbcode-1].weights.z() );
	    scaleFactors[curbcode-1].flag = 0;
	 }
	 
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
	    barcodes.push_back( (*pitr)->barcode() );
	    lastDau++;
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
      //
      if ( HepMC::HEPEVT_Wrapper::number_entries() > nentries ) 
      {
         // yes, need all corrections and adjustments -
	 // figure out how many photons and what particles in 
	 // the decay branch have changes;
	 // also, follow up each one and correct accordingly;
	 // at the same time, add photon(s) to the GenVertex
	 //	 
	 int largestBarcode = -1;
	 int Nbcodes = barcodes.size();
	 for ( int ip=1; ip<Nbcodes; ip++ )
	 {
	    int bcode = barcodes[ip];
	    HepMC::GenParticle* prt = evt->barcode_to_particle( bcode );
	    if ( bcode > largestBarcode ) largestBarcode = bcode;
	    double px = HepMC::HEPEVT_Wrapper::px(ip+1);
	    double py = HepMC::HEPEVT_Wrapper::py(ip+1);
	    double pz = HepMC::HEPEVT_Wrapper::pz(ip+1);
	    double e  = HepMC::HEPEVT_Wrapper::e(ip+1);
	    HepMC::FourVector mom4 = prt->momentum();
	    //double porg = sqrt( mom4.px()*mom4.px() 
	     //                 + mom4.py()*mom4.py() 
		//	      + mom4.pz()*mom4.pz() ) ;
	    //double pnew = sqrt( px*px + py*py + pz*pz );
	    double scale = scaleFactors[bcode-1].weights.x();
	    scaleFactors[bcode-1].weights.setX( scale*(px/mom4.px()) ); 
	    scale = scaleFactors[bcode-1].weights.y();
	    scaleFactors[bcode-1].weights.setY( scale*(py/mom4.py()) ); 
	    scale = scaleFactors[bcode-1].weights.z();
	    scaleFactors[bcode-1].weights.setZ( scale*(pz/mom4.pz()) );
	    scaleFactors[bcode-1].flag = 0; 
	    
	    prt->set_momentum( HepMC::FourVector(px,py,pz,e) );
	    
	    // we do NOT adjust chaldren, etc., here - because we do it 
	    // above, based on whether mother (incoming particles) has
	    // ever been modified
	 
	 }
	 	 
	 int newlyGen =  HepMC::HEPEVT_Wrapper::number_entries() - nentries;
	 
	 if ( largestBarcode < evt->particles_size() )
	 {
	    // need to adjust barcodes down from the afftcted vertex/particles
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
	     // add/shift scale factors towards the end of the list
	     scaleFactors.push_back( Scaling(HepMC::ThreeVector(1.,1.,1.),1) );
	 }  
      } // end of if-statement 

      // ugh, done with this vertex !
   
   }
   
   // restore event number in HEPEVT (safety measure, somehow needed by Hw6)
   HepMC::HEPEVT_Wrapper::set_event_number( evt->event_number() );

   // cross-check printout MODIFIED HepMC::GenEvent
   // evt->print();

   // return conv.read_next_event();
   return evt;
      
}


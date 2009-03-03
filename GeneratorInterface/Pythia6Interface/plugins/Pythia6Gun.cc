/*
 *  $Date: 2009/02/26 11:05:03 $
 *  $Revision: 1.3 $
 *  \author Julia Yarba
 */

#include <iostream>

#include "Pythia6Gun.h"

//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"

using namespace edm;
using namespace gen;


Pythia6Gun::Pythia6Gun( const ParameterSet& pset ) :
   fPy6Service( new Pythia6Service(pset) ),
   fEvt(0)
   // fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") )
{

   // ParameterSet defpset ;
   //ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
   ParameterSet pgun_params = 
      //pset.getUntrackedParameter<ParameterSet>("PGunParameters", defpset ) ;
      pset.getParameter<ParameterSet>("PGunParameters");
      
   // although there's the method ParameterSet::empty(),  
   // it looks like it's NOT even necessary to check if it is,
   // before trying to extract parameters - if it is empty,
   // the default values seem to be taken
   //
   //std::vector<int> defids ;
   //defids.push_back(13) ;
   fPartIDs    = pgun_params.getParameter< std::vector<int> >("ParticleID"); //,defids);  
   fMinEta     = pgun_params.getParameter<double>("MinEta"); // ,-2.2);
   fMaxEta     = pgun_params.getParameter<double>("MaxEta"); // , 2.2);
   fMinPhi     = pgun_params.getParameter<double>("MinPhi"); // ,-3.14159265358979323846);
   fMaxPhi     = pgun_params.getParameter<double>("MaxPhi"); // , 3.14159265358979323846);
   
   fHepMCVerbosity   = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false ) ;
   fPylistVerbosity  = pset.getUntrackedParameter<int>( "pythiaPylistVerbosity", 0 ) ;
   fMaxEventsToPrint = pset.getUntrackedParameter<int>( "maxEventsToPrint", 0 );

   //fAddAntiParticle = pset.getUntrackedParameter("AddAntiParticle", false) ;  

// setup random engine
   randomEngine = &getEngineReference();

// Turn off banner printout
   if (!call_pygive("MSTU(12)=12345")) 
   {
      throw edm::Exception(edm::errors::Configuration,"PythiaError") 
            <<" pythia did not accept MSTU(12)=12345";
   }

   produces<HepMCProduct>();

}

Pythia6Gun::~Pythia6Gun()
{ 
   if ( fPy6Service ) delete fPy6Service; 
   //
   // note that GenEvent or any undelaying (GenVertex, GenParticle) do NOT
   // need to be cleaned, as it'll be done automatically by HepMCProduct
   //
}


void Pythia6Gun::beginJob( const EventSetup& es )
{
   // es.getData( fPDGTable ) ;
   return ;

}

void Pythia6Gun::endJob()
{
}

void Pythia6Gun::beginRun( Run & r, EventSetup const& es )
{

   assert ( fPy6Service ) ;
   
   fPy6Service->setGeneralParams();
   fPy6Service->setCSAParams();
   fPy6Service->setSLHAParams();
      
   call_pyinit("NONE", "", "", 0.0);

   return;
   
}

void Pythia6Gun::endRun( Run & r, EventSetup const& es )
{
   
   // here put in GenRunInfoProduct
   
   call_pystat(1);
   
   return;
}

void Pythia6Gun::attachPy6DecaysToGenEvent()
{

   for ( int iprt=fPartIDs.size(); iprt<pyjets.n; iprt++ ) // the pointer is shifted by -1, c++ style
   {
      int parent = pyjets.k[2][iprt];
      if ( parent != 0 )
      {
         // pull up parent particle
	 //
	 HepMC::GenParticle* parentPart = fEvt->barcode_to_particle( parent );
	 parentPart->set_status( 2 ); // reset status, to mark that it's decayed
	 
	 HepMC::GenVertex* DecVtx = new HepMC::GenVertex(HepMC::FourVector(pyjets.v[0][iprt],
	                                                                   pyjets.v[1][iprt],
					 		                   pyjets.v[2][iprt],
							                   pyjets.v[3][iprt]));
	 DecVtx->add_particle_in( parentPart ); // this will cleanup end_vertex if exists,
	                                        // and replace with the new one
						// I presume barcode will be given automatically
	 
	 // attention: pyjets.k[1][iprt] is PYTHIA6 PID !!!
	 //            need to convert to standard PDG
	 //
	 HepMC::FourVector  pmom(pyjets.p[0][iprt],pyjets.p[1][iprt],
	                         pyjets.p[2][iprt],pyjets.p[3][iprt] );
	 
	 HepMC::GenParticle* daughter = new HepMC::GenParticle(pmom,pyjets.k[1][iprt],1);
	 daughter->suggest_barcode( iprt+1 );
	 DecVtx->add_particle_out( daughter );
	 // give particle barcode as well !

	 int iprt1;
	 for ( iprt1=iprt+1; iprt1<pyjets.n; iprt1++ ) // the pointer is shifted by -1, c++ style
	 {
	    if ( pyjets.k[2][iprt1] != parent ) break; // another parent particle, break the loop
	    HepMC::FourVector  pmomN(pyjets.p[0][iprt1],pyjets.p[1][iprt1],
	                             pyjets.p[2][iprt1],pyjets.p[3][iprt1] );
	    //
	    // same here with PID - need py6->pdg !!!
	    //
	    HepMC::GenParticle* daughterN = new HepMC::GenParticle(pmomN,pyjets.k[1][iprt1],1);
	    daughterN->suggest_barcode( iprt1+1 );
	    DecVtx->add_particle_out( daughterN );	     
	 }
	 
	 iprt = iprt1-1; // reset counter such that it doesn't go over the same child more than once
	                 // don't forget to offset back into c++ counting, as it's already +1 forward

	 fEvt->add_vertex( DecVtx );

      }
   }

   return;

}

void Pythia6Gun::produce( edm::Event& evt, const edm::EventSetup& )
{

   generateEvent() ;
   
   fEvt->set_beam_particles(0,0);
   fEvt->set_event_number(evt.id().event()) ;
   fEvt->set_signal_process_id(pypars.msti[0]) ;  

   attachPy6DecaysToGenEvent();

   if ( evt.id().event() <= fMaxEventsToPrint )
   {
      if ( fPylistVerbosity )
      {
         call_pylist(fPylistVerbosity);
      }
      if ( fHepMCVerbosity )
      {
         if ( fEvt ) fEvt->print();
      }
   }
    
   loadEvent( evt );
}

void Pythia6Gun::loadEvent( edm::Event& evt )
{

   std::auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
   
   if(fEvt)  bare_product->addHepMCData( fEvt );

   evt.put(bare_product);

   
   return;

}

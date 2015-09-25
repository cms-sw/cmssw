/*
 *  \author Julia Yarba
 */

#include <iostream>

#include "Pythia6Gun.h"

//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

using namespace edm;
using namespace gen;


Pythia6Gun::Pythia6Gun( const ParameterSet& pset ) :
   fPy6Service( new Pythia6Service(pset) ),
   fEvt(0)
   // fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") )
{

   ParameterSet pgun_params = 
      pset.getParameter<ParameterSet>("PGunParameters");
      
   // although there's the method ParameterSet::empty(),  
   // it looks like it's NOT even necessary to check if it is,
   // before trying to extract parameters - if it is empty,
   // the default values seem to be taken
   //
   fMinPhi     = pgun_params.getParameter<double>("MinPhi"); // ,-3.14159265358979323846);
   fMaxPhi     = pgun_params.getParameter<double>("MaxPhi"); // , 3.14159265358979323846);
   
   fHepMCVerbosity   = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false ) ;
   fPylistVerbosity  = pset.getUntrackedParameter<int>( "pythiaPylistVerbosity", 0 ) ;
   fMaxEventsToPrint = pset.getUntrackedParameter<int>( "maxEventsToPrint", 0 );

// Turn off banner printout
   if (!call_pygive("MSTU(12)=12345")) 
   {
      throw edm::Exception(edm::errors::Configuration,"PythiaError") 
            <<" pythia did not accept MSTU(12)=12345";
   }

   produces<HepMCProduct>("unsmeared");

}

Pythia6Gun::~Pythia6Gun()
{ 
   if ( fPy6Service ) delete fPy6Service; 
   //
   // note that GenEvent or any undelaying (GenVertex, GenParticle) do NOT
   // need to be cleaned, as it'll be done automatically by HepMCProduct
   //
}


void Pythia6Gun::beginJob()
{
   // es.getData( fPDGTable ) ;
   return ;

}

void Pythia6Gun::beginRun( Run const&, EventSetup const& es )
{
   std::cout << " FYI: MSTU(10)=1 is ENFORCED in Py6-PGuns, for technical reasons"
             << std::endl;
   return;
}

void Pythia6Gun::beginLuminosityBlock(LuminosityBlock const& lumi, EventSetup const&) {

   assert ( fPy6Service ) ;

   RandomEngineSentry<Pythia6Service> sentry(fPy6Service, lumi.index());

   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance

   fPy6Service->setGeneralParams();
   fPy6Service->setCSAParams();
   fPy6Service->setSLHAParams();
   
   call_pygive("MSTU(10)=1");
      
   call_pyinit("NONE", "", "", 0.0);
}

void Pythia6Gun::endRun( Run const&, EventSetup const& es )
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
	 
	 HepMC::FourVector  pmom(pyjets.p[0][iprt],pyjets.p[1][iprt],
	                         pyjets.p[2][iprt],pyjets.p[3][iprt] );
	 
	 int dstatus = 0;
	 if ( pyjets.k[0][iprt] >= 1 && pyjets.k[0][iprt] <= 10 )  
	 {
	    dstatus = 1;
	 }
	 else if ( pyjets.k[0][iprt] >= 11 && pyjets.k[0][iprt] <= 20 ) 
	 {
	    dstatus = 2;
	 }
	 else if ( pyjets.k[0][iprt] >= 21 && pyjets.k[0][iprt] <= 30 ) 
	 {
	    dstatus = 3;
	 }
	 else if ( pyjets.k[0][iprt] >= 31 && pyjets.k[0][iprt] <= 100 )
	 {
	    dstatus = pyjets.k[0][iprt];
	 }
	 HepMC::GenParticle* daughter = 
	    new HepMC::GenParticle(pmom,
	                           HepPID::translatePythiatoPDT( pyjets.k[1][iprt] ),
				   dstatus);
	 daughter->suggest_barcode( iprt+1 );
	 DecVtx->add_particle_out( daughter );
	 // give particle barcode as well !

	 int iprt1;
	 for ( iprt1=iprt+1; iprt1<pyjets.n; iprt1++ ) // the pointer is shifted by -1, c++ style
	 {
	    if ( pyjets.k[2][iprt1] != parent ) break; // another parent particle, break the loop

	    HepMC::FourVector  pmomN(pyjets.p[0][iprt1],pyjets.p[1][iprt1],
	                             pyjets.p[2][iprt1],pyjets.p[3][iprt1] );

	    dstatus = 0;
	    if ( pyjets.k[0][iprt1] >= 1 && pyjets.k[0][iprt1] <= 10 )  
	    {
	       dstatus = 1;
	    }
	    else if ( pyjets.k[0][iprt1] >= 11 && pyjets.k[0][iprt1] <= 20 ) 
	    {
	       dstatus = 2;
	    }
	    else if ( pyjets.k[0][iprt1] >= 21 && pyjets.k[0][iprt1] <= 30 ) 
	    {
	       dstatus = 3;
	    }
	    else if ( pyjets.k[0][iprt1] >= 31 && pyjets.k[0][iprt1] <= 100 )
	    {
	       dstatus = pyjets.k[0][iprt1];
	    }
	    HepMC::GenParticle* daughterN = 
	       new HepMC::GenParticle(pmomN,
	                              HepPID::translatePythiatoPDT( pyjets.k[1][iprt1] ),
				      dstatus);
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
   RandomEngineSentry<Pythia6Service> sentry(fPy6Service, evt.streamID());

   generateEvent(fPy6Service->randomEngine()) ;

   fEvt->set_beam_particles(0,0);
   fEvt->set_event_number(evt.id().event()) ;
   fEvt->set_signal_process_id(pypars.msti[0]) ;  

   attachPy6DecaysToGenEvent();

   int evtN = evt.id().event();
   if ( evtN <= fMaxEventsToPrint )
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

   evt.put(bare_product, "unsmeared");

   
   return;

}

HepMC::GenParticle* Pythia6Gun::addAntiParticle( int& ip, int& particleID,
                                                 double& ee, double& eta, double& phi )
{

   if ( ip < 2 ) return 0;

// translate PDG to Py6
   int py6PID = HepPID::translatePDTtoPythia( particleID );
// Check if particle is its own anti-particle.
   int pythiaCode = pycomp_(py6PID); // this is py6 internal validity check, it takes Pythia6 pid
	                             // so actually I'll need to convert
   int has_antipart = pydat2.kchg[3-1][pythiaCode-1];
   int particleID2 = has_antipart ? -1 * particleID : particleID; // this is PDG, for HepMC::GenEvent
   int py6PID2 = has_antipart ? -1 * py6PID : py6PID;	 // this py6 id, for py1ent   
   double the = 2.*atan(exp(eta));
   phi  = phi + M_PI;
   if (phi > 2.* M_PI) {phi = phi - 2.* M_PI;}         

   // copy over mass of the previous one, because then py6 will pick it up
   pyjets.p[4][ip-1] = pyjets.p[4][ip-2];

   py1ent_(ip, py6PID2, ee, the, phi);

   double px     = pyjets.p[0][ip-1]; // pt*cos(phi) ;
   double py     = pyjets.p[1][ip-1]; // pt*sin(phi) ;
   double pz     = pyjets.p[2][ip-1]; // mom*cos(the) ;
   HepMC::FourVector ap(px,py,pz,ee) ;
   HepMC::GenParticle* APart =
	       new HepMC::GenParticle(ap,particleID2,1);
   APart->suggest_barcode( ip ) ;

   return APart;

} 

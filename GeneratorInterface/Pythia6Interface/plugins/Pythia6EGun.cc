
#include <iostream>

#include "Pythia6EGun.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace gen;

Pythia6EGun::Pythia6EGun( const ParameterSet& pset ) :
   Pythia6Gun(pset)
{
   
   ParameterSet defpset ;
   //ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
   ParameterSet pgun_params = 
      pset.getUntrackedParameter<ParameterSet>("PGunParameters", defpset ) ;
   fMinE       = pgun_params.getUntrackedParameter<double>("MinE",  0.);
   fMaxE       = pgun_params.getUntrackedParameter<double>("MaxE",  0.);
   fAddAntiParticle = pgun_params.getUntrackedParameter("AddAntiParticle", false) ;  

}

Pythia6EGun::~Pythia6EGun()
{
}

void Pythia6EGun::produce( Event& evt, const edm::EventSetup& )
{
   
   std::auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  

   // now actualy, start cooking up the event gun 
   //

   // 1st, primary vertex
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.));

   // here re-create fEvt (memory)
   //
   fEvt = new HepMC::GenEvent() ;
     
   int ip=1;
   for ( size_t i=0; i<fPartIDs.size(); i++ )
   {
	 int particleID = fPartIDs[i]; // this is PDG - need to convert to Py6 !!!
	 double pi = 3.1415927;
         int dum = 0;
	 double ee=0,the=0,eta=0;
	 double mass = pymass_(particleID);
	 double phi = (fMaxPhi-fMinPhi)*pyr_(&dum)+fMinPhi;
	 ee   = (fMaxE-fMinE)*pyr_(&dum)+fMinE;
	 eta  = (fMaxEta-fMinEta)*pyr_(&dum)+fMinEta;                                                      
	 the  = 2.*atan(exp(-eta));                                                                          
	 
	 phi = phi * (3.1415927/180.);

	 py1ent_(ip, particleID, ee, the, phi);
	 
	 // fill up also HepMC::GenEvent
         double mom2   = ee*ee - mass*mass ;
         double mom    = 0. ;
         if (mom2 > 0.) 
         {
            mom = sqrt(mom2) ;
         }
         else
         {
            mom = 0. ;
         }
         double px     = mom*sin(the)*cos(phi) ;
         double py     = mom*sin(the)*sin(phi) ;
         double pz     = mom*cos(the) ;
         
	 HepMC::FourVector p(px,py,pz,ee) ;
         HepMC::GenParticle* Part = 
             new HepMC::GenParticle(p,particleID,1);
         Part->suggest_barcode( ip ) ;
         Vtx->add_particle_out(Part);
	 
	 if(fAddAntiParticle)
	 {
	    ip = ip + 1;
// Check if particle is its own anti-particle.
            int pythiaCode = pycomp_(particleID); // this is py6 internal validity check, it takes Pythia6 pid
	                                          // so actually I'll need to convert
            int has_antipart = pydat2.kchg[3-1][pythiaCode-1];
            int particleID2 = has_antipart ? -1 * particleID : particleID;	    
	    the = 2.*atan(exp(eta));
	    phi  = phi + 3.1415927;
	    if (phi > 2.* 3.1415927) {phi = phi - 2.* 3.1415927;}         
	    py1ent_(ip, particleID2, ee, the, phi);
            HepMC::FourVector ap(-px,-py,-pz,ee) ;
	    HepMC::GenParticle* APart =
	       new HepMC::GenParticle(ap,particleID2,1);
	    APart->suggest_barcode( ip ) ;
	    Vtx->add_particle_out(APart) ;	    
	 }
	 ip++;
   }
   
   fEvt->add_vertex(Vtx);
     
   // run pythia
   pyexec_();
   
   // here compose py6 decays into HepMC::GenEvent "by hands"
   //
   // no need to clean up GenEvent memory - done in HepMCProduct
   // here re-create fEvt (memory)
   //
   fEvt->set_beam_particles(0,0);
   fEvt->set_event_number(evt.id().event()) ;
   fEvt->set_signal_process_id(pypars.msti[0]) ;  
   fEvt->set_event_scale(pypars.pari[16]);

/*
   // loop over pyjets.n
   //
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
	    HepMC::GenParticle* daughterN = new HepMC::GenParticle(pmomN,pyjets.k[1][iprt1],1);
	    daughterN->suggest_barcode( iprt1+1 );
	    DecVtx->add_particle_out( daughterN );	     
	 }
	 
	 iprt = iprt1-1; // reset counter such that it doesn't go over the same child more than once
	                 // don't forget to offset back into c++ counting, as it's already +1 forward

	 fEvt->add_vertex( DecVtx );

      }
   }
*/

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
   
   if(fEvt)  bare_product->addHepMCData( fEvt );

   evt.put(bare_product);

   return;
}

DEFINE_FWK_MODULE(Pythia6EGun);

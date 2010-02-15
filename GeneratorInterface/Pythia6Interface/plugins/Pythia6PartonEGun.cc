
#include <iostream>

#include "Pythia6PartonEGun.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace gen;

Pythia6PartonEGun::Pythia6PartonEGun( const ParameterSet& pset ) :
   Pythia6PartonGun(pset)
{
   
   // ParameterSet defpset ;
   ParameterSet pgun_params = 
      pset.getParameter<ParameterSet>("PGunParameters"); //, defpset ) ;
   fMinEta     = pgun_params.getParameter<double>("MinEta"); // ,-2.2);
   fMaxEta     = pgun_params.getParameter<double>("MaxEta"); // , 2.2);
   fMinE       = pgun_params.getParameter<double>("MinE"); // ,  20.);
   fMaxE       = pgun_params.getParameter<double>("MaxE"); // , 420.);

}

Pythia6PartonEGun::~Pythia6PartonEGun()
{
}

void Pythia6PartonEGun::generateEvent()
{
   
   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance

   // now actualy, start cooking up the event gun 
   //

   // 1st, primary vertex
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.));

   // here re-create fEvt (memory)
   //
   fEvt = new HepMC::GenEvent() ;
     
   int ip=1;
   
   int py6PID = HepPID::translatePDTtoPythia( fPartonID );
   int dum = 0;
   double ee=0, the=0, eta=0;
   double mass = pymass_(py6PID);
	 
   // fill p(ip,5) (in PYJETS) with mass value right now, 
   // because the (hardcoded) mstu(10)=1 will make py1ent
   // pick the mass from there
   pyjets.p[4][ip-1]=mass; 
	 	 
   double phi = (fMaxPhi-fMinPhi)*pyr_(&dum)+fMinPhi;   
   ee   = (fMaxE-fMinE)*pyr_(&dum)+fMinE;
   eta  = (fMaxEta-fMinEta)*pyr_(&dum)+fMinEta;                                                      
   the  = 2.*atan(exp(-eta));                                                                          
	 
   py1ent_(ip, py6PID, ee, the, phi);
	 
   double px     = pyjets.p[0][ip-1]; // pt*cos(phi) ;
   double py     = pyjets.p[1][ip-1]; // pt*sin(phi) ;
   double pz     = pyjets.p[2][ip-1]; // mom*cos(the) ;
         
   HepMC::FourVector p(px,py,pz,ee) ;
   HepMC::GenParticle* Part = new HepMC::GenParticle(p,fPartonID,1);
   Part->suggest_barcode( ip ) ;
   Vtx->add_particle_out(Part);
	 
   // now add anti-quark
   ip = ip + 1;

// Check if there's an "anti"
   int pythiaCode = pycomp_(py6PID); // this is py6 internal validity check, it takes Pythia6 pid
	                             // so actually I'll need to convert
   int has_antipart = pydat2.kchg[3-1][pythiaCode-1];
   int partonID2 = has_antipart ? -1 * fPartonID : fPartonID;
   int py6PID2 = has_antipart ? -1 * py6PID : py6PID;	 // this py6 id, for py1ent    
   the = 2.*atan(exp(eta));
   phi  = phi + M_PI;
   if (phi > 2.* M_PI) {phi = phi - 2.* M_PI;}  
	    
   // same trick as above
   pyjets.p[4][ip-1] = mass;
	           
   py1ent_(ip, py6PID2, ee, the, phi);
	    
   HepMC::FourVector ap(-px,-py,-pz,ee) ;
   HepMC::GenParticle* APart = new HepMC::GenParticle(ap,partonID2,1);
   
   APart->suggest_barcode( ip ) ;
   Vtx->add_particle_out(APart) ;	    

   // this should probably be configurable...
   //
   double qmax = 2.*ee;
   
   joinPartons( qmax );
         
   fEvt->add_vertex(Vtx);
     
   // run pythia
   pyexec_();
   
   return;
}

DEFINE_FWK_MODULE(Pythia6PartonEGun);

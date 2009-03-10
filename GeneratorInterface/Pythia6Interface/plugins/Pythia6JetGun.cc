
#include <iostream>

#include "Pythia6JetGun.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace gen;

Pythia6JetGun::Pythia6JetGun( const ParameterSet& pset ) :
   Pythia6Gun(pset),
   fMinE(0.),
   fMaxE(0.),
   fMinP(0.),
   fMaxP(0.)
{
   
   // ParameterSet defpset ;
   ParameterSet pgun_params = 
      pset.getParameter<ParameterSet>("PGunParameters"); 
   
   fMinE = pgun_params.getParameter<double>("MinE");     
   fMaxE = pgun_params.getParameter<double>("MaxE");     
   fMinP = pgun_params.getParameter<double>("MinP"); 
   fMaxP = pgun_params.getParameter<double>("MaxP"); 

}

Pythia6JetGun::~Pythia6JetGun()
{
}

void Pythia6JetGun::generateEvent()
{
   
   // now actualy, start cooking up the event gun 
   //

   // 1st, primary vertex
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.));

   // here re-create fEvt (memory)
   //
   fEvt = new HepMC::GenEvent() ;
     
   int ip=1;
   double totPx = 0.;
   double totPy = 0.;
   double totPz = 0.;
   double totE  = 0.;
   double totM  = 0.;
   double phi, eta, the, ee, pp;
   int dum = 0;
   for ( size_t i=0; i<fPartIDs.size(); i++ )
   {
	 int particleID = fPartIDs[i]; // this is PDG - need to convert to Py6 !!!
	 	 
	 // internal numbers
	 //
	 phi = 2. * M_PI * pyr_(&dum);
	 the = std::acos( -1. + 2.*pyr_(&dum) );
	 
	 // from input
	 //
	 ee   = (fMaxE-fMinE)*pyr_(&dum)+fMinE;
	 
	 // add entry to py6
	 //
	 py1ent_(ip, particleID, ee, the, phi);

         // values for computing total mass
	 //
         totPx += pyjets.p[0][ip-1];
	 totPy += pyjets.p[1][ip-1];
	 totPz += pyjets.p[2][ip-1];
	 totE  += pyjets.p[3][ip-1];
	 
	 ip++;
	 
   } // end forming up py6 record of the jet

	 
   // compute total mass
   //   
   totM = std::sqrt( totE*totE - (totPx*totPx+totPy*totPy+totPz*totPz) );
	 
   //now the boost (from input params)
   //
   pp = (fMaxP-fMinP)*pyr_(&dum)+fMinP; 
   ee = std::sqrt( totM*totM + pp*pp );	 
	 
   //the boost direction (from input params)
   //
   phi = (fMaxPhi-fMinPhi)*pyr_(&dum)+fMinPhi;
   eta  = (fMaxEta-fMinEta)*pyr_(&dum)+fMinEta;                                                      
   the  = 2.*atan(exp(-eta));  
	 
   double betaX = pp/ee * std::sin(the) * std::cos(phi);
   double betaY = pp/ee * std::sin(the) * std::sin(phi);
   double betaZ = pp/ee * std::cos(the);  
   
   // boost all particles
   // the first 2 params (-1) tell to boost all (fisrt-to-last),
   // and the next 2 params (0.) tell no rotation
   // 
   int first=-1, last=-1;
   double rothe=0, rophi=0.;
   
   pyrobo_( first, last, rothe, rophi, betaX, betaY, betaZ );                                                                      
	          
   // event should be formed from boosted record !!!
   // that's why additional loop
   //
   for ( int i=0; i<pyjets.n; i++ )
   {
      HepMC::FourVector p(pyjets.p[0][i],pyjets.p[1][i],pyjets.p[2][i],pyjets.p[3][i]) ;
      //
      // here we pass pythia6 particle ID, not PDG - FIXME !!!
      //
      HepMC::GenParticle* Part = 
             new HepMC::GenParticle(p,pyjets.k[1][i],1); 
      Part->suggest_barcode( i+1 ) ;
      Vtx->add_particle_out(Part);
   }
   fEvt->add_vertex(Vtx);
     
   // run pythia
   pyexec_();
   
   double totalE = 0.;
   for ( int i=0; i<pyjets.n; i++ )
   {
      if ( pyjets.k[0][i] == 1 )
      {
         totalE += pyjets.p[3][i];
      }
   }
   
   return;
}

DEFINE_FWK_MODULE(Pythia6JetGun);

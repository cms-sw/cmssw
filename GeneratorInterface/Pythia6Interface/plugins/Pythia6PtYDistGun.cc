
#include <iostream>

#include "Pythia6PtYDistGun.h"
#include "GeneratorInterface/Pythia6Interface/interface/PtYDistributor.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace gen;

Pythia6PtYDistGun::Pythia6PtYDistGun( const ParameterSet& pset ) :
   Pythia6Gun(pset), fPtYGenerator(0)
{
   
   ParameterSet pgun_params = 
      pset.getParameter<ParameterSet>("PGunParameters"); 
   
   fPtYGenerator = new PtYDistributor( pgun_params.getParameter<FileInPath>("kinematicsFile"), 
                                       getEngineReference(), 
				       pgun_params.getParameter<double>("MaxPt"),
                                       pgun_params.getParameter<double>("MinPt"),
				       pgun_params.getParameter<double>("MaxY"), 
				       pgun_params.getParameter<double>("MinY"), 
				       pgun_params.getParameter<int>("PtBinning"), 
				       pgun_params.getParameter<int>("YBinning") );
}

Pythia6PtYDistGun::~Pythia6PtYDistGun()
{
   if ( fPtYGenerator ) delete fPtYGenerator;
}

void Pythia6PtYDistGun::generateEvent()
{
   
   // now actualy, start cooking up the event gun 
   //

   // 1st, primary vertex
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.) );

   // here re-create fEvt (memory)
   //
   fEvt = new HepMC::GenEvent() ;
     
   int ip=1;

   for ( size_t i=0; i<fPartIDs.size(); i++ )
   {

	 int particleID = fPartIDs[i]; // this is PDG - need to convert to Py6 !!!
         int dum = 0;
	 double pt=0, y=0, u=0, ee=0, the=0;
	 
	 pt = fPtYGenerator->firePt();
	 y  = fPtYGenerator->fireY();
	 u  = exp(y); 
	 
	 double mass = pymass_(particleID);
	 
	 ee = 0.5*std::sqrt(mass*mass+pt*pt)*(u*u+1)/u;
	 
	 double pz = std::sqrt(ee*ee-pt*pt-mass*mass);
	 if ( y < 0. ) pz *= -1;
	 
	 double phi = (fMaxPhi-fMinPhi)*pyr_(&dum)+fMinPhi;

	 the  = atan(pt/pz);
	 if ( pz < 0. ) the += M_PI ;                                                                      
	 
	 py1ent_(ip, particleID, ee, the, phi);
	 
         double px     = pt*cos(phi) ;
         double py     = pt*sin(phi) ;
         
	 HepMC::FourVector p(px,py,pz,ee) ;
         HepMC::GenParticle* Part = 
             new HepMC::GenParticle(p,particleID,1);
         Part->suggest_barcode( ip ) ;
         Vtx->add_particle_out(Part);
	 
	 ip++;
   
   }
  
   fEvt->add_vertex(Vtx);
     
   // run pythia
   pyexec_();
   
   return;
}

DEFINE_FWK_MODULE(Pythia6PtYDistGun);

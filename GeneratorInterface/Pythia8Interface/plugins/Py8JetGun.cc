
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"
#include "GeneratorInterface/Pythia8Interface/interface/RandomP8.h"

namespace gen {

class Py8JetGun : public Py8GunBase {
   
   public:
      
      Py8JetGun( edm::ParameterSet const& );
      ~Py8JetGun() {}
	 
      bool generatePartonsAndHadronize();
      const char* classname() const;
	 
   private:
      
      // PtGun particle(s) characteristics
      double  fMinEta;
      double  fMaxEta;
      double  fMinP ;
      double  fMaxP ;
      double  fMinE ;
      double  fMaxE ;

};

// implementation 
//
Py8JetGun::Py8JetGun( edm::ParameterSet const& ps )
   : Py8GunBase(ps) 
{

   // ParameterSet defpset ;
   edm::ParameterSet pgun_params = 
      ps.getParameter<edm::ParameterSet>("PGunParameters"); // , defpset ) ;
   fMinEta     = pgun_params.getParameter<double>("MinEta"); // ,-2.2);
   fMaxEta     = pgun_params.getParameter<double>("MaxEta"); // , 2.2);
   fMinP       = pgun_params.getParameter<double>("MinP"); // ,  0.);
   fMaxP       = pgun_params.getParameter<double>("MaxP"); // ,  0.);
   fMinE       = pgun_params.getParameter<double>("MinE"); // ,  0.);
   fMaxE       = pgun_params.getParameter<double>("MaxE"); // ,  0.);

}

bool Py8JetGun::generatePartonsAndHadronize()
{

   fMasterGen->event.reset();

   double totPx = 0.;
   double totPy = 0.;
   double totPz = 0.;
   double totE  = 0.;
   double totM  = 0.;
   double phi, eta, the, ee, pp;
   
   for ( size_t i=0; i<fPartIDs.size(); i++ )
   {

      int particleID = fPartIDs[i]; // this is PDG - need to convert to Py8 ???

      // FIXME !!!
      // Ouch, it's using bare randomEngine pointer - that's NOT safe.
      // Need to hold a pointer somewhere properly !!!
      //
      phi = 2. * M_PI * randomEngine->flat() ;
      the = acos( -1. + 2.*randomEngine->flat() );

      // from input
      //
      ee   = (fMaxE-fMinE)*randomEngine->flat() + fMinE;
            
      double mass = (fMasterGen->particleData).m0( particleID );

      pp = sqrt( ee*ee - mass*mass );
      
      double px = pp * sin(the) * cos(phi);
      double py = pp * sin(the) * sin(phi);
      double pz = pp * cos(the);

      if ( !((fMasterGen->particleData).isParticle( particleID )) )
      {
         particleID = std::fabs(particleID) ;
      }
      (fMasterGen->event).append( particleID, 1, 0, 0, px, py, pz, ee, mass ); 
      
      // values for computing total mass
      //
      totPx += px;
      totPy += py;
      totPz += pz;
      totE  += ee;

   }

   totM = sqrt( totE*totE - (totPx*totPx+totPy*totPy+totPz*totPz) );

   //now the boost (from input params)
   //
   pp = (fMaxP-fMinP)*randomEngine->flat() + fMinP; 
   ee = sqrt( totM*totM + pp*pp );	 

   //the boost direction (from input params)
   //
   phi = (fMaxPhi-fMinPhi)*randomEngine->flat() + fMinPhi;
   eta  = (fMaxEta-fMinEta)*randomEngine->flat() + fMinEta;                                                      
   the  = 2.*atan(exp(-eta));  

   double betaX = pp/ee * std::sin(the) * std::cos(phi);
   double betaY = pp/ee * std::sin(the) * std::sin(phi);
   double betaZ = pp/ee * std::cos(the);  

   // boost all particles
   //   
   (fMasterGen->event).bst( betaX, betaY, betaZ );
   
   if ( !fMasterGen->next() ) return false;
   
   event().reset(new HepMC::GenEvent);
   return toHepMC.fill_next_event( fMasterGen->event, event().get() );
  
}

const char* Py8JetGun::classname() const
{
   return "Py8JetGun"; 
}

typedef edm::GeneratorFilter<gen::Py8JetGun, gen::ExternalDecayDriver> Pythia8JetGun;

} // end namespace

using gen::Pythia8JetGun;
DEFINE_FWK_MODULE(Pythia8JetGun);

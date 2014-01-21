
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"
#include "GeneratorInterface/Pythia8Interface/interface/RandomP8.h"

namespace gen {

class Py8PtGun : public Py8GunBase {
   
   public:
      
      Py8PtGun( edm::ParameterSet const& );
      ~Py8PtGun() {}
	 
      bool generatePartonsAndHadronize();
      const char* classname() const;
	 
   private:
      
      // PtGun particle(s) characteristics
      double  fMinEta;
      double  fMaxEta;
      double  fMinPt ;
      double  fMaxPt ;
      bool    fAddAntiParticle;

};

// implementation 
//
Py8PtGun::Py8PtGun( edm::ParameterSet const& ps )
   : Py8GunBase(ps) 
{

   // ParameterSet defpset ;
   edm::ParameterSet pgun_params = 
      ps.getParameter<edm::ParameterSet>("PGunParameters"); // , defpset ) ;
   fMinEta     = pgun_params.getParameter<double>("MinEta"); // ,-2.2);
   fMaxEta     = pgun_params.getParameter<double>("MaxEta"); // , 2.2);
   fMinPt      = pgun_params.getParameter<double>("MinPt"); // ,  0.);
   fMaxPt      = pgun_params.getParameter<double>("MaxPt"); // ,  0.);
   fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle"); //, false) ;  

}

bool Py8PtGun::generatePartonsAndHadronize()
{

   fMasterGen->event.reset();
   
   for ( size_t i=0; i<fPartIDs.size(); i++ )
   {

      int particleID = fPartIDs[i]; // this is PDG - need to convert to Py8 ???

      // FIXME !!!
      // Ouch, it's using bare randomEngine pointer - that's NOT safe.
      // Need to hold a pointer somewhere properly !!!
      //
      double phi = (fMaxPhi-fMinPhi) * randomEngine->flat() + fMinPhi;
      double eta  = (fMaxEta-fMinEta) * randomEngine->flat() + fMinEta;                                                      
      double the  = 2.*atan(exp(-eta));                                                                          

      double pt   = (fMaxPt-fMinPt) * randomEngine->flat() + fMinPt;
      
      double mass = (fMasterGen->particleData).m0( particleID );

      double pp = pt / sin(the); // sqrt( ee*ee - mass*mass );
      double ee = sqrt( pp*pp + mass*mass );
      
      double px = pt * cos(phi);
      double py = pt * sin(phi);
      double pz = pp * cos(the);

      if ( !((fMasterGen->particleData).isParticle( particleID )) )
      {
         particleID = std::fabs(particleID) ;
      }
      (fMasterGen->event).append( particleID, 1, 0, 0, px, py, pz, ee, mass ); 

// Here also need to add anti-particle (if any)
// otherwise just add a 2nd particle of the same type 
// (for example, gamma)
//
      if ( fAddAntiParticle )
      {
         if ( (fMasterGen->particleData).isParticle( -particleID ) )
	 {
	    (fMasterGen->event).append( -particleID, 1, 0, 0, px, py, pz, ee, mass );
	 }
	 else
	 {
	    (fMasterGen->event).append( particleID, 1, 0, 0, px, py, pz, ee, mass );
	 }
      }

   }
   
   if ( !fMasterGen->next() ) return false;
   
   event().reset(new HepMC::GenEvent);
   return toHepMC.fill_next_event( fMasterGen->event, event().get() );
  
}

const char* Py8PtGun::classname() const
{
   return "Py8PtGun"; 
}

typedef edm::GeneratorFilter<gen::Py8PtGun, gen::ExternalDecayDriver> Pythia8PtGun;

} // end namespace

using gen::Pythia8PtGun;
DEFINE_FWK_MODULE(Pythia8PtGun);

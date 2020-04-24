#ifndef gen_ExternalDecayDriver_h
#define gen_ExternalDecayDriver_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <string>
#include <vector>

namespace HepMC
{
class GenEvent;
}

namespace CLHEP {
   class HepRandomEngine;
}

namespace lhef { 
  class LHEEvent;
}

namespace gen {

  class EvtGenInterfaceBase;
  class TauolaInterfaceBase;
  class PhotosInterfaceBase;


   class ExternalDecayDriver{
      public:
         
	 // ctor & dtor
	 ExternalDecayDriver( const edm::ParameterSet& );
	 ~ExternalDecayDriver();
	 
	 void init( const edm::EventSetup& );

	 const std::vector<int>&         operatesOnParticles() { return fPDGs; }
	 const std::vector<std::string>& specialSettings()     { return fSpecialSettings; }
	 
	 HepMC::GenEvent* decay( HepMC::GenEvent* evt);
	 HepMC::GenEvent* decay( HepMC::GenEvent* evt, lhef::LHEEvent *lheEvent);
 
	 void statistics() const;

         void setRandomEngine(CLHEP::HepRandomEngine*);
         std::vector<std::string> const& sharedResources() const { return exSharedResources; }

      private:
	 bool                     fIsInitialized;
	 TauolaInterfaceBase*     fTauolaInterface;
	 EvtGenInterfaceBase*     fEvtGenInterface;
	 PhotosInterfaceBase*     fPhotosInterface;
	 std::vector<int>         fPDGs;
	 std::vector<std::string> fSpecialSettings;

         std::vector<std::string> exSharedResources;
   };

}

#endif

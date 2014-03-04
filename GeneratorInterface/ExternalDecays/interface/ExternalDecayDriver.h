#ifndef gen_ExternalDecayDriver_h
#define gen_ExternalDecayDriver_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace HepMC
{
class GenEvent;
}

namespace gen {

  class EvtGenInterfaceBase;
  class TauolaInterfaceBase;
  class PhotosInterfaceBase;

   class ExternalDecayDriver 
   {
      public:
	 // ctor & dtor
	 ExternalDecayDriver( const edm::ParameterSet& );
	 ~ExternalDecayDriver();
	 
	 void init( const edm::EventSetup& );
	 const std::vector<int>&         operatesOnParticles() { return fPDGs; }
	 const std::vector<std::string>& specialSettings()     { return fSpecialSettings; }
	 HepMC::GenEvent* decay( HepMC::GenEvent* );
	 void statistics() const;
      
      private:
	 bool                     fIsInitialized;
	 TauolaInterfaceBase*     fTauolaInterface;
	 EvtGenInterfaceBase*     fEvtGenInterface;
	 PhotosInterfaceBase*     fPhotosInterface;
	 std::vector<int>         fPDGs;
	 std::vector<std::string> fSpecialSettings;
   };
}

#endif

#ifndef gen_ExternalDecays_TauolaInterface_h
#define gen_ExternalDecays_TauolaInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace HepMC 
{
class GenEvent;
}


namespace gen {

   class TauolaInterface
   {
      public:
      
      // ctor & dtor
      TauolaInterface( const edm::ParameterSet& );
      ~TauolaInterface();
      
      void init();
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( const HepMC::GenEvent* );
      void statistics() ;
      
      private: 
      
      //            
      std::vector<int> fPDGs;
      int              fPolarization;
       
   };

}

#endif

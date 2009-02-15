#ifndef gen_EvtGenInterface_h
#define gen_EvtGenInterface_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace HepMC
{
class GenEvent;
}

namespace gen {

   class EvtGenInterface
   {
      public:
      
      // ctor & dtor
      EvtGenInterface( const edm::ParameterSet& );
      ~EvtGenInterface();
      
      bool decay( HepMC::GenEvent* );
      
      private: 
      
       
   };

}

#endif

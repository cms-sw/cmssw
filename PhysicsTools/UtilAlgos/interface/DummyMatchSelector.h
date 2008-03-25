#ifndef DummyMatchSelector_h
#define DummyMatchSelector_h

/**
   Dummy class for preselection of object matches.
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {

  template<typename T1, typename T2>
  class DummyMatchSelector {
  
    public:
    
      DummyMatchSelector(const edm::ParameterSet& cfg) {  }
      
      bool operator()( const T1 & c, const T2 & hlt ) const { return true; }
      
  };
  
}


#endif

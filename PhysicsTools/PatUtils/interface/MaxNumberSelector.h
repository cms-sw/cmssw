#ifndef PhysicsTools_PatAlgos_MaxNumberSelector_h
#define PhysicsTools_PatAlgos_MaxNumberSelector_h

#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"


struct MaxNumberSelector {
  MaxNumberSelector( unsigned int maxNumber ) : 
    maxNumber_( maxNumber ) { }
  bool operator()( unsigned int number ) const { return number <= maxNumber_; }

private:
  unsigned int maxNumber_;
};


namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MaxNumberSelector> {
      static MaxNumberSelector make( const edm::ParameterSet & cfg ) {
	return MaxNumberSelector( cfg.getParameter<unsigned int>( "maxNumber" ) );
      }
    };

  }
}

#endif

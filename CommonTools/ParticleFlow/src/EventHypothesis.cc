#include "CommonTools/ParticleFlow/interface/EventHypothesis.h"

#include <iostream>

using pf2pat::EventHypothesis; 


EventHypothesis::EventHypothesis( const edm::ParameterSet& ps) {
  
  std::cout<<ps.dump()<<std::endl;
}


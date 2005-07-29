#ifndef PHYSICSTOOLS_TWOSAMEBODYCOMBINER_H
#define PHYSICSTOOLS_TWOSAMEBODYCOMBINER_H
// $Id: TwoSameBodyCombiner.h,v 1.1 2005/07/14 11:45:30 llista Exp $
#include "PhysicsTools/CandAlgos/src/TwoBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace phystools {
  
  class TwoSameBodyCombiner : public TwoBodyCombiner {
  public:
    TwoSameBodyCombiner( const edm::ParameterSet & );
  private:
    void produce( edm::Event& e, const edm::EventSetup& );
    std::string source;
  };
}

#endif

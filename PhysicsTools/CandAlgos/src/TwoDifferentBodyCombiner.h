#ifndef PHYSICSTOOLS_TWODIFFERENTBODYCOMBINER_H
#define PHYSICSTOOLS_TWODIFFERENTBODYCOMBINER_H
// $Id: TwoDifferentBodyCombiner.h,v 1.1 2005/07/14 11:45:30 llista Exp $
#include "PhysicsTools/CandAlgos/src/TwoBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace phystools {
  
  class TwoDifferentBodyCombiner : public TwoBodyCombiner {
  public:
    TwoDifferentBodyCombiner( const edm::ParameterSet & );
  private:
    void produce( edm::Event& e, const edm::EventSetup& );
    std::string source1, source2;
  };
}

#endif

#ifndef FIXED_PU_GENERATOR_H
#define FIXED_PU_GENERATOR_H

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/
#include "Mixing/Base/interface/PUGenerator.h"

namespace edm 
{

  class FixedPUGenerator: public PUGenerator
    {
    public:
      explicit FixedPUGenerator(int average): nrEvtsPerBunch(average) { }
      ~FixedPUGenerator() { }
    
    private:
      virtual unsigned int numberOfEventsPerBunch() const { return nrEvtsPerBunch;}

      unsigned int nrEvtsPerBunch;
    };
}//edm

#endif 

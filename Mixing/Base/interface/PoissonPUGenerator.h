#ifndef POISSON_PU_GENERATOR_H
#define POISSON_PU_GENERATOR_H

#include "Mixing/Base/interface/PUGenerator.h"
#include "CLHEP/Random/RandPoissonQ.h"

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm 
{

  class PoissonPUGenerator: public PUGenerator
    {
    public:
      explicit PoissonPUGenerator(double av) :average(av){ }
      ~PoissonPUGenerator() { }
    
    private:
      virtual unsigned int numberOfEventsPerBunch() const {return CLHEP::RandPoissonQ::fire(average);}
      double average;
    };
}//edm

#endif 

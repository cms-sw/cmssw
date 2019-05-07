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
      double average;
      virtual unsigned int numberOfEventsPerBunch() const {return CLHEP::RandPoissonQ::fire(this.average);}
    };
}//edm

#endif 

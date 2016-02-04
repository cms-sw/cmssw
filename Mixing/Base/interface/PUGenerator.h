#ifndef PU_GENERATOR_H
#define PU_GENERATOR_H

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm
{

  class PUGenerator
    {
    public:
      explicit PUGenerator() { }
      virtual ~PUGenerator() { }
      virtual unsigned int numberOfEventsPerBunch() const=0;
    
    private:
    };

}//edm
#endif 

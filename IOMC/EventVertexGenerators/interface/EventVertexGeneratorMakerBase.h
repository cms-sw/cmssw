#ifndef IOMC_EventVertexGeneratorMakerBase_h
#define IOMC_EventVertexGeneratorMakerBase_h

#include <memory>

namespace edm{
  class ParameterSet;
}

class EventVertexGeneratorMakerBase
{
   public:
      EventVertexGeneratorMakerBase() {}
      virtual ~EventVertexGeneratorMakerBase() {}
      virtual std::auto_ptr<BaseEventVertexGenerator> 
         make( const edm::ParameterSet&) = 0 ;
};

#endif

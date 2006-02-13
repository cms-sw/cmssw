#ifndef IOMC_EventVertexGeneratorMakerBase_h
#define IOMC_EventVertexGeneratorMakerBase_h

#include <memory>

// class SimActivityRegistry;
namespace edm{
  class ParameterSet;
}

class EventVertexGeneratorMakerBase
{
   public:
      EventVertexGeneratorMakerBase() {}
      virtual ~EventVertexGeneratorMakerBase() {}
      //virtual std::auto_ptr<BaseEventVertexGenerator> make(const edm::ParameterSet&,
      //					      SimActivityRegistry&) const = 0;
      virtual std::auto_ptr<BaseEventVertexGenerator> make(const edm::ParameterSet& ) = 0 ;
};

#endif

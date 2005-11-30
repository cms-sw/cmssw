#ifndef IOMC_EventVertexGeneratorMaker_h
#define IOMC_EventVertexGeneratorMaker_h

#include <memory>

#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"

template<class T>
class EventVertexGeneratorMaker : public EventVertexGeneratorMakerBase
{
   public:
      EventVertexGeneratorMaker(){}
      virtual std::auto_ptr<BaseEventVertexGenerator> make(const edm::ParameterSet& p,
                                SimActivityRegistry& reg) const
      {
	std::auto_ptr<T> returnValue(new T(p));
	SimActivityRegistryEnroller::enroll(reg, returnValue.get());
	return std::auto_ptr<BaseEventVertexGenerator>(returnValue);
      }
};

#endif

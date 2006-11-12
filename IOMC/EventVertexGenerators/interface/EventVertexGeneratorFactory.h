#ifndef IOMC_EventVertexGeneratorFactory_H
#define IOMC_EventVertexGeneratorFactory_H

#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"
#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorMaker.h"

#include "SealKernel/Component.h"
#include "PluginManager/PluginFactory.h"

class EventVertexGeneratorFactory 
    : public seal::PluginFactory<
    EventVertexGeneratorMakerBase *() >
{
public:
    virtual ~EventVertexGeneratorFactory();
    static EventVertexGeneratorFactory * get(); 
private:
    static EventVertexGeneratorFactory s_instance;
    EventVertexGeneratorFactory();
};
//NOTE: the prefix "IOMC/EventVertexGenerators/" is there for 'backwards compatability
// and should eventually be removed (which will require changes to config files)
#define DEFINE_EVENTVERTEXGENERATOR(type) \
  DEFINE_SEAL_PLUGIN(EventVertexGeneratorFactory, EventVertexGeneratorMaker<type>,"IOMC/EventVertexGenerators/" #type)

#endif

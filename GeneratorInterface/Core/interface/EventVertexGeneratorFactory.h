#ifndef GeneratorInterface_Core_EventVertexGeneratorFactory_h
#define GeneratorInterface_Core_EventVertexGeneratorFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

class BaseEvtVtxGenerator;

namespace edm {
  class ConsumesCollector;
  class ParameterSet;

  typedef BaseEvtVtxGenerator*(DAFunc)(ParameterSet const&, ConsumesCollector&);
  typedef edmplugin::PluginFactory<DAFunc> EventVertexGeneratorPluginFactory;

  class EventVertexGeneratorFactory {
  public:
    ~EventVertexGeneratorFactory();

    static EventVertexGeneratorFactory const* get();

    std::unique_ptr<BaseEvtVtxGenerator>
      makeEventVertexGenerator(ParameterSet const&, ConsumesCollector&) const;

  private:
    EventVertexGeneratorFactory();
    static EventVertexGeneratorFactory const singleInstance_;
  };
}

#define DEFINE_EVENTVERTEX_GENERATOR(type) \
  DEFINE_EDM_PLUGIN (edm::EventVertexGeneratorPluginFactory,type,#type)
  //DEFINE_EDM_PLUGIN (edm::EventVertexGeneratorPluginFactory,type,#type); DEFINE_FWK_PSET_DESC_FILLER(type)

#endif

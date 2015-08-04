
#include "GeneratorInterface/Core/interface/EventVertexGeneratorFactory.h"
#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::EventVertexGeneratorPluginFactory,"EventVertexGenerator");

namespace edm {
  namespace one {
    class EDProducerBase;
  }

  EventVertexGeneratorFactory::~EventVertexGeneratorFactory() {
  }

  EventVertexGeneratorFactory::EventVertexGeneratorFactory() {
  }

  EventVertexGeneratorFactory const EventVertexGeneratorFactory::singleInstance_;

  EventVertexGeneratorFactory const* EventVertexGeneratorFactory::get() {
    // will not work with plugin factories
    //static EventVertexGeneratorFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::unique_ptr<BaseEvtVtxGenerator>
  EventVertexGeneratorFactory::makeEventVertexGenerator(ParameterSet const& conf, ConsumesCollector& iC) const {
    std::string vertexGeneratorType = conf.getParameter<std::string>("vertexGeneratorType");
    FDEBUG(1) << "EventVertexGeneratorFactory: digi_accumulator_type = " << vertexGeneratorType << std::endl;
    std::unique_ptr<BaseEvtVtxGenerator> wm;
    wm = std::unique_ptr<BaseEvtVtxGenerator>(EventVertexGeneratorPluginFactory::get()->create(vertexGeneratorType, conf, iC));
    
    if(wm.get()==0) {
	throw edm::Exception(errors::Configuration,"NoSourceModule")
	  << "EventVertexGenerator Factory:\n"
	  << "Cannot find dig type from ParameterSet: "
	  << vertexGeneratorType << "\n"
	  << "Perhaps your source type is misspelled or is not an EDM Plugin?\n"
	  << "Try running EdmPluginDump to obtain a list of available Plugins.";
    }

    FDEBUG(1) << "EventVertexGeneratorFactory: created a BaseEvtVtxGenerator"
	      << vertexGeneratorType
	      << std::endl;

    return wm;
  }
}

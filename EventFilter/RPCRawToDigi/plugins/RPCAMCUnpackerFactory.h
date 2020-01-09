#ifndef EventFilter_RPCRawToDigi_RPCAMCUnpackerFactory_h
#define EventFilter_RPCRawToDigi_RPCAMCUnpackerFactory_h

#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
}  // namespace edm

class RPCAMCUnpacker;

typedef edmplugin::PluginFactory<RPCAMCUnpacker *(edm::ParameterSet const &, edm::ProducesCollector)>
    RPCAMCUnpackerFactory;

#endif  // EventFilter_RPCRawToDigi_RPCAMCUnpackerFactory_h

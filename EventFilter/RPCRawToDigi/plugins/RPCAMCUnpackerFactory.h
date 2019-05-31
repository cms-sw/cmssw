#ifndef EventFilter_RPCRawToDigi_RPCAMCUnpackerFactory_h
#define EventFilter_RPCRawToDigi_RPCAMCUnpackerFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
  namespace stream {
    class EDProducerBase;
  }  // namespace stream
}  // namespace edm

class RPCAMCUnpacker;

typedef edmplugin::PluginFactory<RPCAMCUnpacker *(edm::stream::EDProducerBase &producer, edm::ParameterSet const &)>
    RPCAMCUnpackerFactory;

#endif  // EventFilter_RPCRawToDigi_RPCAMCUnpackerFactory_h

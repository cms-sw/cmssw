#ifndef CondCore_DBCommon_BlobStreamerPluginFactory_h
#define CondCore_DBCommon_BlobStreamerPluginFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondCore/ORA/interface/IBlobStreamingService.h"
#include <string>

namespace cond {
  typedef edmplugin::PluginFactory< ora::IBlobStreamingService*() > BlobStreamerPluginFactory;
}

#endif

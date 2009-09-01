#ifndef CondCore_DBCommon_BlobStreamerPluginFactory_h
#define CondCore_DBCommon_BlobStreamerPluginFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "POOLCore/IBlobStreamingService.h"
#include <string>

namespace cond {
  typedef edmplugin::PluginFactory< pool::IBlobStreamingService*(const std::string&) > BlobStreamerPluginFactory;
}

#endif

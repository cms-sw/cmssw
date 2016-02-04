#ifndef CondCore_DBCommon_TechnologyProxyFactory_h
#define CondCore_DBCommon_TechnologyProxyFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <memory>
#include <string>
namespace cond{
  typedef edmplugin::PluginFactory< cond::TechnologyProxy*() > TechnologyProxyFactory;
}
#endif

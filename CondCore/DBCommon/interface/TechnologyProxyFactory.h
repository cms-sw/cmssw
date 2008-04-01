#ifndef CondCore_DBCommon_TechnologyProxyFactory_h
#define CondCore_DBCommon_TechnologyProxyFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
namespace cond{
  typedef edmplugin::PluginFactory< cond::TechnologyProxy*(const std::string&) > TechnologyProxyFactory;
}
EDM_REGISTER_PLUGINFACTORY(cond::TechnologyProxyFactory,"DBTechnologyPlugin");
#endif

#include "CoralKernel/Service.h"
#include "CoralKernel/ILoadableComponent.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "CondCore/CondDB/interface/CoralServiceManager.h"
#include "CondCore/CondDB/interface/CoralServiceFactory.h"
coral::ILoadableComponent* 
cond::CoralServiceManager::newComponent( const std::string& componentname ){
  return cond::CoralServiceFactory::get()->create(componentname);
}

std::set<std::string> 
cond::CoralServiceManager::knownPlugins() const{
  std::vector<edmplugin::PluginInfo> pinfo=cond::CoralServicePluginFactory::get()->available();
  std::set<std::string> r;
  std::vector<edmplugin::PluginInfo>::iterator i;
  std::vector<edmplugin::PluginInfo>::iterator ibeg=pinfo.begin();
  std::vector<edmplugin::PluginInfo>::iterator iend=pinfo.end();
  for(i=ibeg;i<iend;++i){
    r.insert(i->name_);
  }
  return r;
}

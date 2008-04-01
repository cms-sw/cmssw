#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  const char* path = getenv("LD_LIBRARY_PATH");
  std::string spath(path? path: "");
  std::string::size_type last=0;
  std::string::size_type i=0;
  std::vector<std::string> paths;
  while( (i=spath.find_first_of(':',last))!=std::string::npos) {
    paths.push_back(spath.substr(last,i-last));
    last = i+1;
    std::cout <<paths.back()<<std::endl;
  }
  paths.push_back(spath.substr(last,std::string::npos));
  config.searchPath(paths);
  edmplugin::PluginManager::configure(config);
  std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create("sqlite"));
  std::cout<<ptr->getRealConnectString("hello")<<std::endl;
}

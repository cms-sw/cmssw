#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <iostream>
int main(){
  /* edmplugin::PluginManager::Config config;
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
  */
  std::cout<<"testing Connection Handler "<<std::endl;
  cond::DBSession* session=new cond::DBSession;
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  conHandler.registerConnection("sqlite_file:mydata.db",*session,0);
  conHandler.registerConnection("sqlite_fip:CondCore/SQLiteData/data/mydata.db",*session,0);
  session->open();
  conHandler.connect(session);
  conHandler.disconnectAll();
  std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create("sqlite","sqlite_file:pippo.db"));
  std::cout<<ptr->getRealConnectString()<<std::endl;
  static cond::ConnectionHandler& conHandler2=cond::ConnectionHandler::Instance();
  conHandler.registerConnection("frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_PRESH",*session,0);
  std::auto_ptr<cond::TechnologyProxy> ptr2(cond::TechnologyProxyFactory::get()->create("frontier","frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_PRESH"));
  std::cout<<ptr2->getRealConnectString()<<std::endl;
  delete session;
}

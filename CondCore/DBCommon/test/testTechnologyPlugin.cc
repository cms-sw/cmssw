#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include <iostream>


int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type",std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(services);
  
  std::cout<<"testing Connection Handler "<<std::endl;
  cond::DbConnection conn;
  conn.configure( cond::CmsDefaults );
  cond::DbSession session = conn.createSession();
  const char * connects[] = {
    "sqlite_file:mydata.db",
    "sqlite_fip:CondCore/SQLiteData/data/mydata.db",
    "frontier://FrontierDev/CMS_COND_PRESH",
    "frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_PRESH"
  };
  for (int i=0; i<4; ++i) {
    try {
      session.open(connects[i]);
      std::cout << connects[i] << " " <<  session.connectionString() << std::endl;
    } catch ( const cond::Exception & er) {
      std::cout << "error " << er.what();
    }
  }

  return 0;
}


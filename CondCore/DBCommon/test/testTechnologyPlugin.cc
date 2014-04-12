#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include <iostream>

struct Connections {
  const char * conStr;
  bool readOnly;
};

int main(){
  try {
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
    conn.configuration().setAuthenticationPath("/afs/cern.ch/cms/DB/conddb");
    conn.configuration().setMessageLevel( coral::Debug );
    conn.configure();
    cond::DbSession session = conn.createSession();
  
    const Connections connects[] = {
      {"sqlite_file:technologyPlugin.db", false},
      {"oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS", true},
      //{"sqlite_fip:CondCore/SQLiteData/data/technologyPlugin.db", true},
      {"frontier://FrontierDev/CMS_COND_UNIT_TESTS", true}, 
      {"frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_UNIT_TESTS", true}
    };
    for (int i=0; i<4; ++i) {
      session.open(connects[i].conStr,connects[i].readOnly);
      std::cout << connects[i].conStr << " " <<  session.connectionString() << (connects[i].readOnly ? " in read mode": " in read-write mode") << std::endl;
    }
  } catch ( const std::exception & er) {
    std::cout << "Error: " << er.what()<<std::endl;
  } 
  return 0;
}


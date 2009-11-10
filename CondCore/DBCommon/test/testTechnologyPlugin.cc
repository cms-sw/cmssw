#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::cout<<"testing Connection Handler "<<std::endl;
  cond::DbConnection conn;
  conn.configure( cond::CmsDefaults );
  cond::DbSession session = conn.createSession();
  const char * connects[] = {
    "sqlite_file:mydata.db",
    "sqlite_fip:CondCore/SQLiteData/data/mydata.db",
    "frontier:FrontierDev/CMS_COND_PRESH",
    "frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_PRESH"
  };
  for (int i=0; int<4; ++i) {
    try {
      session.open(connects[i]);
      std::cout << connects[i] << " " <<  session.connectionString() << std::endl;
    } catch ( const cond::Exception & er) {
      std::cout << "error " << er.what();
    }
  }

  return 0;
}


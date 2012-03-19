#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/AttributeSpecification.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection* conn = new cond::DbConnection;
  conn->configure( cond::CmsDefaults );
  std::string connStr("sqlite_file:testDbScopedTransaction.db");
  std::string tok0("");
  try {
    cond::DbSession s0 = conn->createSession();
    delete conn;
    s0.open(  connStr );
    s0.transaction().start();
    std::cout << "Transaction active at beginning A="<<s0.transaction().isActive()<<std::endl;
    {
      cond::DbScopedTransaction tr0(s0);
      // this should have no effect...
      std::cout << "Sc.Transaction active at A0.0="<<tr0.isActive()<<std::endl;
      std::cout << "Transaction active at A0.0="<<s0.transaction().isActive()<<std::endl;
    }
    {
      cond::DbScopedTransaction tr1(s0);
      tr1.start();
      std::cout << "Sc.Transaction active at A1.0="<<tr1.isActive()<<std::endl;
      std::cout << "Transaction active at A1.0="<<s0.transaction().isActive()<<std::endl;
      tr1.commit();
    }
    std::cout << "Transaction active at A1.1="<<s0.transaction().isActive()<<std::endl;
    s0.transaction().commit();
    std::cout << "Transaction active at end A="<<s0.transaction().isActive()<<std::endl;
    s0.transaction().start();
    std::cout << "Transaction active at beginning B="<<s0.transaction().isActive()<<std::endl;
    {
      cond::DbScopedTransaction tr2(s0);
      tr2.start();
      std::cout << "Sc.Transaction active at B1.0="<<tr2.isActive()<<std::endl;
      std::cout << "Transaction active at B1.0="<<s0.transaction().isActive()<<std::endl;
      // no commit causes rollback...
    }
    std::cout << "Transaction active at B1.1="<<s0.transaction().isActive()<<std::endl;
    try {
      s0.transaction().start();
      std::cout << "Transaction active at beginning C="<<s0.transaction().isActive()<<std::endl;
      {
        cond::DbScopedTransaction tr3(s0);
        tr3.start();
        std::cout << "Sc.Transaction active at C1.0="<<tr3.isActive()<<std::endl;
        std::cout << "Transaction active at C1.0="<<s0.transaction().isActive()<<std::endl;
        // throw exception...
        throw cond::Exception("Transaction interrupted...");
        tr3.commit();
      }
      std::cout << "Transaction still active at C1.1="<<s0.transaction().isActive()<<std::endl;
    } catch ( const cond::Exception& exc ){
      std::cout << "Expected exception: "<<exc.what()<<std::endl;
      std::cout << "Transaction active at C1.1="<<s0.transaction().isActive()<<std::endl;      
    }
    std::cout << "Transaction active at very end="<<s0.transaction().isActive()<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "Expected error: "<<exc.what()<<std::endl;
  }
  
}

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
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
  std::string connStr("sqlite_file:testDbTransaction.db");
  std::string tok0("");
  try {
    cond::DbSession s0 = conn->createSession();
    delete conn;
    s0.open(  connStr );
    std::cout << "Transaction active at beginning="<<s0.transaction().isActive()<<std::endl;
    bool rb0 = s0.transaction().rollback();
    if(!rb0) std::cout << "Rollback not performed, since transaction is inactive..."<<std::endl;
    else std::cout << "ERROR: rollback performed on inactive transaction."<<std::endl;
    int nt0 = s0.transaction().start();
    std::cout << "Transaction open at 0.0="<<nt0<<std::endl;
    std::cout << "Transaction active at 0.0="<<s0.transaction().isActive()<<std::endl;
    coral::ISchema& schema = s0.nominalSchema();
    schema.dropIfExistsTable( "mytest" );
    coral::TableDescription description0;
    description0.setName( "mytest" );
    description0.insertColumn( "ID",coral::AttributeSpecification::typeNameForId( typeid(int) ) );
    description0.insertColumn( "X",coral::AttributeSpecification::typeNameForId( typeid(float) ) );
    description0.insertColumn( "Y",coral::AttributeSpecification::typeNameForId( typeid(float) ) );
    description0.insertColumn( "ORDER",coral::AttributeSpecification::typeNameForId( typeid(int) ) );
    schema.createTable( description0 );
    {
      cond::DbSession s1 = s0;
      int nt1 = s1.transaction().start();
      s1.createDatabase();
      std::cout << "Transaction open at 1.0="<<nt1<<std::endl;
      std::cout << "Transaction active at 1.0="<<s0.transaction().isActive()<<std::endl;
      boost::shared_ptr<int> data( new int(100) );
      tok0 = s1.storeObject( data.get(),"cont0");
      std::cout << "Stored object with id = "<<tok0<<std::endl;
      nt1 = s1.transaction().commit();
      std::cout << "Transaction still open at 1.1="<<nt1<<std::endl;
    }
    int nt2 = s0.transaction().start( true );
    std::cout << "Transaction open at 2.0="<<nt2<<std::endl;
    std::cout << "Transaction active at 2.0="<<s0.transaction().isActive()<<std::endl;
    nt0 = s0.transaction().commit();
    std::cout << "Transaction still open at 0.1="<<nt0<<std::endl;
    std::cout << "Transaction active at end="<<s0.transaction().isActive()<<std::endl;
    int nt3 = s0.transaction().start( true );
    std::cout << "Transaction open at 3.0="<<nt3<<std::endl;
    std::cout << "Transaction active at 3.0="<<s0.transaction().isActive()<<std::endl;
    int nt4 = s0.transaction().start( true );
    std::cout << "Transaction open at 4.0="<<nt4<<std::endl;
    std::cout << "Transaction active at 4.0="<<s0.transaction().isActive()<<std::endl;
    bool rb1 = s0.transaction().rollback();
    if(!rb1) std::cout << "ERROR, Rollback not performed, and transaction is active..."<<std::endl;
    else std::cout << "Rollback performed on active transaction."<<std::endl;
    std::cout << "Transaction active at 4.1="<<s0.transaction().isActive()<<std::endl;
    nt4 = s0.transaction().commit();
    std::cout << "Transaction still open at 4.2="<<nt4<<std::endl;
    std::cout << "Transaction active at end="<<s0.transaction().isActive()<<std::endl;
    s0.close();
    s0.open(  connStr );
    bool fc0 = s0.transaction().forceCommit();
    if(fc0) std::cout << "ERROR: transaction cannot be forced to commit, because it has been not started."<<std::endl;
    else     std::cout << "Transaction is not active: "<<s0.transaction().isActive()<<std::endl;
    s0.transaction().start( true );
    s0.transaction().start( true );
    s0.transaction().start( true );
    s0.transaction().start( true );
    int nt5 = s0.transaction().start( true );
    std::cout << "Transaction open at 5.0="<<nt5<<std::endl;
    std::cout << "Transaction active at 5.0="<<s0.transaction().isActive()<<std::endl;
    bool fc1 = s0.transaction().forceCommit();
    if(fc1) std::cout << "Transaction has been committed and is now not active: "<<s0.transaction().isActive()<<std::endl;
    else std::cout << "ERROR: transaction has not been forced to commit"<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "Expected error: "<<exc.what()<<std::endl;
  }
  
}

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/TableDescription.h"
#include "CoralBase/AttributeSpecification.h"
#include "DataSvc/Ref.h"
#include <iostream>
int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::DbConnection* conn = new cond::DbConnection;
  conn->configure( cond::CmsDefaults );
  std::string connStr("sqlite_file:mytest.db");
  {
    cond::DbSession session = conn->createSession();
    session.close();
    try {
      session.nominalSchema();
      std::cout << "ERROR: expected exception not thrown (0)"<<std::endl;
    } catch ( const cond::Exception& exc ){
      std::cout << "Expected error: "<<exc.what()<<std::endl;
    }
    try {
      session.transaction().start();
      std::cout << "ERROR: expected exception not thrown (1)"<<std::endl;
    } catch ( const cond::Exception& exc ){
      std::cout << "Expected error: "<<exc.what()<<std::endl;
    }  
    session.open( connStr );
    std::cout << "Session successfully open on: "<<connStr<<std::endl;
    session.close();
    session.open( connStr );
  }
  cond::DbSession s;
  if(s.isOpen()){
    std::cout << "ERROR: s should not be open yet..."<<std::endl;
  }
  
  {
    cond::DbSession session = conn->createSession();
    session.open( connStr );
    s = session;
    if(!s.isOpen()){
      std::cout << "ERROR: s should be open now..."<<std::endl;
    }
  }
  try {
    s.nominalSchema();
    std::cout << "Session is correctly open."<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "ERROR: "<<exc.what()<<std::endl;
  }
  s.close();
  cond::DbSession s2 = s;
  s2.open( connStr );
  try {
    s2.nominalSchema();
    std::cout << "Session is correctly open."<<std::endl;
    s2 = s;
    s2.transaction().start();
    s2.nominalSchema();
    s2.transaction().commit();    
  } catch ( const cond::Exception& exc ){
    std::cout << "ERROR: "<<exc.what()<<std::endl;
  }
  // closing connection...
  cond::DbConnection conn2 = *conn;
  delete conn;
  conn2.close();
  try {
    s2.transaction().start();
    std::cout << "ERROR: expected exception not thrown (1)"<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "Expected error 1: "<<exc.what()<<std::endl;
  }
  try {
    s2.nominalSchema();
    std::cout << "ERROR: expected exception not thrown (2)"<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "Expected error 2: "<<exc.what()<<std::endl;
  }
  try {
    s2.poolCache();
    std::cout << "ERROR: expected exception not thrown (3)"<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "Expected error 3: "<<exc.what()<<std::endl;
  }
  try {
    s2.getObject(std::string(""));
    std::cout << "ERROR: expected exception not thrown (4)"<<std::endl;
  } catch ( const cond::Exception& exc ){
    std::cout << "Expected error 4: "<<exc.what()<<std::endl;
  }
  
}

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
  std::string connStr0("sqlite_file:mytest.db");
  std::string connStr1("sqlite_file:mytest1.db");
  cond::DbSession s0;
  {
    cond::DbSession session = conn->createSession();
    session.open( connStr0 );
    s0 = session;
    if(!s0.isOpen()){
      std::cout << "ERROR: s0 should be open now..."<<std::endl;
    }
  }
  cond::DbSession s1 = conn->createSession();
  delete conn;
  s1.open( connStr1 );
  std::string tok0("");
  std::string tok1("");
  try {
    s0.transaction().start();
    coral::ISchema& schema = s0.nominalSchema();
    schema.dropIfExistsTable( "mytest" );
    coral::TableDescription description0;
    description0.setName( "mytest" );
    description0.insertColumn( "ID",coral::AttributeSpecification::typeNameForId( typeid(int) ) );
    description0.insertColumn( "X",coral::AttributeSpecification::typeNameForId( typeid(float) ) );
    description0.insertColumn( "Y",coral::AttributeSpecification::typeNameForId( typeid(float) ) );
    description0.insertColumn( "ORDER",coral::AttributeSpecification::typeNameForId( typeid(int) ) );
    schema.createTable( description0 );
    pool::Ref<int> myRef0 = s0.storeObject(new int(100),"cont0");
    tok0 = myRef0.toString();
    std::cout << "Stored object with id = "<<tok0<<std::endl;
    s0.transaction().commit();
    s0.close();
    s1.transaction().start();
    pool::Ref<std::string> myRef1 = s1.storeObject<std::string>(new std::string("blabla"),"cont1");
    tok1 = myRef1.toString();
    std::cout << "Stored object with id = "<<tok1<<std::endl;
    s1.transaction().commit();
    s1.close();
    s0.open( connStr0 );
    s0.transaction().start(true);
    coral::ISchema& rschema = s0.nominalSchema();
    std::set<std::string> result=rschema.listTables();
    for(std::set<std::string>::iterator it=result.begin(); it!=result.end(); ++it){
      std::cout<<"table names: "<<*it<<std::endl;
    }
    pool::Ref<int> readRef0 = s0.getTypedObject<int>( tok0 );
    std::cout << "Object with id="<<tok0<<" has been read with value="<<*readRef0<<std::endl;
    s0.transaction().commit();     
    s1.open( connStr1 );
    s1.transaction().start(true);
    pool::Ref<std::string> readRef1= s1.getTypedObject<std::string>( tok1 );
    std::cout << "Object with id="<<tok1<<" has been read with value="<<*readRef1<<std::endl;
    s1.transaction().commit();
    s0.transaction().start();
    s1.transaction().start(true);
    std::string tok2 = s0.importObject( s1, tok1 );
    std::cout << "Object with id="<<tok1<<" imported with id="<<tok2<<std::endl;
    s1.transaction().commit();
    s0.transaction().commit();
    s0.close();
    s0.open( connStr0 );
    s0.transaction().start(true);
    pool::Ref<std::string> readRef2= s0.getTypedObject<std::string>( tok2 );
    std::cout << "Object with id="<<tok2<<" has been read with value="<<*readRef2<<std::endl;
    s0.transaction().commit();    
  } catch ( const cond::Exception& exc ){
    std::cout << "ERROR: "<<exc.what()<<std::endl;
  } 
}

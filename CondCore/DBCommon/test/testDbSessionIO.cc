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
  std::string connStr0("sqlite_file:testDbSessionIO0.db");
  std::string connStr1("sqlite_file:testDbSessionIO1.db");
  cond::DbSession s0;
  {
    std::cout << "######### test 0"<<std::endl;
    cond::DbSession session = conn->createSession();
    session.open( connStr0 );
    s0 = session;
    if(!s0.isOpen()){
      std::cout << "ERROR: s0 should be open now..."<<std::endl;
    }
  }
    std::cout << "######### test 1"<<std::endl;
  cond::DbSession s1 = conn->createSession();
  delete conn;
  s1.open( connStr1 );
  std::string tok0("");
  std::string tok1("");
  try {
    std::cout << "######### test 2"<<std::endl;
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
    std::cout << "######### test 3"<<std::endl;
    boost::shared_ptr<int> data( new int(100) );
    s0.createDatabase();
    tok0 = s0.storeObject( data.get(),"cont0");
    std::cout << "Stored object with id = "<<tok0<<std::endl;
    s0.transaction().commit();
    s0.close();
    s1.transaction().start();
    std::cout << "######### test 4"<<std::endl;
    boost::shared_ptr<std::string> sdata( new std::string("blabla") );
    s1.createDatabase();
    tok1 = s1.storeObject<std::string>(sdata.get(),"cont1");
    std::cout << "Stored object with id = "<<tok1<<std::endl;
    s1.transaction().commit();
    s1.close();
    s0.open( connStr0 );
    s0.transaction().start(true);
    std::cout << "######### test 5"<<std::endl;
    coral::ISchema& rschema = s0.nominalSchema();
    std::set<std::string> result=rschema.listTables();
    for(std::set<std::string>::iterator it=result.begin(); it!=result.end(); ++it){
      std::cout<<"table names: "<<*it<<std::endl;
    }
    std::cout << "######### test 6"<<std::endl;
    boost::shared_ptr<int> read0 = s0.getTypedObject<int>( tok0 );
    std::cout << "Object with id="<<tok0<<" has been read with value="<<*read0<<std::endl;
    s0.transaction().commit();     
    s1.open( connStr1 );
    s1.transaction().start(true);
    std::cout << "######### test 7"<<std::endl;
    boost::shared_ptr<std::string> read1= s1.getTypedObject<std::string>( tok1 );
    std::cout << "Object with id="<<tok1<<" has been read with value="<<*read1<<std::endl;
    s1.transaction().commit();
    s0.transaction().start();
    s1.transaction().start(true);
    std::cout << "######### test 8"<<std::endl;
    std::string tok2 = s0.importObject( s1, tok1 );
    std::cout << "Object with id="<<tok1<<" imported with id="<<tok2<<std::endl;
    s1.transaction().commit();
    s0.transaction().commit();
    s0.close();
    s0.open( connStr0 );
    std::cout << "######### test 9"<<std::endl;
    s0.transaction().start(true);
    boost::shared_ptr<std::string> read2= s0.getTypedObject<std::string>( tok2 );
    std::cout << "Object with id="<<tok2<<" has been read with value="<<*read2<<std::endl;
    s0.transaction().commit();    
  } catch ( const cond::Exception& exc ){
    std::cout << "ERROR: "<<exc.what()<<std::endl;
  } 
}

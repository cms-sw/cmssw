#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/test/TestBase.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
namespace {

  void testDbWRFunc( ora::Database& db ){
    ora::Object obj;
    std::string data("BlaBla");
    std::string contName("MyCont");
    try {
      db.create();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.drop();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.createContainer<std::string>( contName );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.createContainer<std::string>();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.dropContainer( contName );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.insertItem( contName, obj );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.insert( contName, data );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.insert( data );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
  }
  
  void testDbUPFunc( ora::Database& db ){
    ora::OId oid;
    ora::Object obj;
    std::string data("BlaBla");
    try {
      db.updateItem( oid, obj );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.update( oid, data );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.erase( oid );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
  }
  
  void testDbRDFunction( ora::Database& db ){
    try {
      db.exists();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.containers();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    std::string contName( "NewCont");
    try {
      db.containerHandle( contName );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.containerHandle( 1 );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    ora::OId oid;
    try {
      db.fetchItem( oid );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.fetch<std::string>( oid );
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
  }
  
  void testAllFunction( ora::Database& db ){
    testDbWRFunc( db );
    testDbUPFunc( db );
    testDbRDFunction( db );
    try {
      db.transaction();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
    try {
      db.flush();
    } catch ( const ora::Exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    } catch ( const std::exception& e ){
      std::cout << "* Error: "<<e.what()<<std::endl;
    }
  }
}

class TestORABasicException : public ora::TestBase {
  public:
    TestORABasicException(): 
      TestBase( "testORABasicException" ){
    }

    virtual ~TestORABasicException(){
    }

    int execute( const std::string& connStr ){
      ora::Database db;
      std::cout <<"####### Case0: Database NOT connected."<<std::endl; 
      bool connected = db.isConnected();
      std::string okConn("");
      if( !connected ) okConn = "NOT ";
      std::cout << "# Database is "<<okConn<<" connected.Connstr=["<<db.connectionString()<<"]"<<std::endl;
      testAllFunction( db );
      db.disconnect();
      std::cout <<"####### Case1: Connect to an invalid database."<<std::endl; 
      std::string invalidConnection("invalid");
      try{
	db.connect( invalidConnection );
      } catch ( const ora::Exception& e ){
	std::cout << "# Expected Error: "<<e.what()<<std::endl;
      } catch ( const std::exception& e ){
	std::cout << "# Expected Error: "<<e.what()<<std::endl;
      } 
      std::cout <<"####### Case2: Connect to a valid database. Transaction not started."<<std::endl; 
      db.connect( connStr );
      connected = db.isConnected();
      okConn = std::string("");
      if( !connected ) okConn = "NOT ";
      std::cout << "# Database is now "<<okConn<<" connected.Connstr=["<<db.connectionString()<<"]"<<std::endl;
      testAllFunction( db );
      std::cout <<"####### Case3: ReadOnly Transaction started."<<std::endl; 
      db.transaction().start();
      testDbWRFunc( db );
      testDbUPFunc( db );
      std::cout << "# committing" << std::endl;
      db.transaction().commit();
      std::cout << "# disconnecting..."<<std::endl;
      db.disconnect();
      sleep();
      std::cout <<"####### Case4: Update Transaction started."<<std::endl; 
      db.connect( connStr );
      db.transaction().start( false );
      std::cout << "# Transaction started\n# Dropping database" << std::endl;
      if(db.exists()) db.drop();
      std::cout << "# Database dropped\n# Committing transaction" << std::endl;
      db.transaction().commit();
      std::cout << "# Transaction committed\n# Closing session" << std::endl;
      db.disconnect();
      std::cout << "# Session closed" << std::endl;
      return 0;
    }
};

int main(int argc, char** argv){
  TestORABasicException test;
  return test.run();
}


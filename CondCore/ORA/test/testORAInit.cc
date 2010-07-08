#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(){
  std::string authpath("/afs/cern.ch/cms/DB/conddb");
  std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
  ::putenv(const_cast<char*>(pathenv.c_str()));
  
  std::string connStr( "oracle://cms_orcoff_prep/CMS_COND_WEB" );
  try {
    ora::Database db;
    //std::string connStr( "sqlite_file:test.db" );
    db.connect( connStr );
    db.transaction().start( false );
    if( db.exists() ) db.drop();
    db.transaction().commit();
    db.disconnect();
  }catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  ora::OId id0, id1, id2, id3;
  try {
    ora::Database db;
    db.configuration().properties().setFlag( ora::Configuration::automaticDatabaseCreation() );
    //std::string connStr( "sqlite_file:test.db" );
    db.connect( connStr );
    db.transaction().start( false );
    int data = 99;
    id0 = db.insert( data );
    std::cout << "** inserted data contid="<<id0.containerId()<<" oid="<<id0.itemId()<<std::endl;
    std::string s("ORATest1234567890");
    id1 = db.insert( s );
    std::cout << "** inserted data contid="<<id1.containerId()<<" oid="<<id1.itemId()<<std::endl;
    id2 = db.insert( 1000 );
    std::cout << "** inserted data contid="<<id2.containerId()<<" oid="<<id2.itemId()<<std::endl;
    db.flush();
    db.transaction().commit();
  }catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  ::sleep(1);
  try {
    ora::Database db;
    db.connect( connStr );
    db.transaction().start();
    boost::shared_ptr< int > r0 = db.fetch< int >( id0 );
    std::cout << "** read data contid="<<id0.containerId()<<" oid="<<id0.itemId()<<" data="<<*r0<<std::endl;
    boost::shared_ptr< std::string > r1 = db.fetch< std::string >( id1 );
    std::cout << "** read data contid="<<id1.containerId()<<" oid="<<id1.itemId()<<" data="<<*r1<<std::endl;
    db.transaction().commit();
  }catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  try {
    ora::Database db;
    //std::string connStr( "sqlite_file:test.db" );
    db.connect( connStr );
    db.transaction().start( false );
    if( db.exists() ) db.drop();
    db.transaction().commit();
    db.disconnect();
  }catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  try {
    ora::Database db;
    db.configuration().properties().setFlag( ora::Configuration::automaticContainerCreation() );
    db.configuration().setMessageVerbosity( coral::Debug );
    db.connect( connStr );
    db.transaction().start( false );
    int data = 99;
    try {
      id0 = db.insert( data ); 
    } catch ( ora::Exception& e ){
      std::cout << "# Expected exception:"<<e.what()<<std::endl;
    }
    if( !db.exists() ){
      std::cout << "# Db does not exists, creating it..."<<std::endl;
      db.create();
    }
    id0 = db.insert( data ); 
    std::cout << "** inserted data contid="<<id0.containerId()<<" oid="<<id0.itemId()<<std::endl;
    std::string s("ORATest1234567890");
    id1 = db.insert( s );
    std::cout << "** inserted data contid="<<id1.containerId()<<" oid="<<id1.itemId()<<std::endl;
    id2 = db.insert( 1000 );
    std::cout << "** inserted data contid="<<id2.containerId()<<" oid="<<id2.itemId()<<std::endl;
    db.flush();
    db.transaction().commit();
  }catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  ::sleep(1);
  try {
    ora::Database db;
    db.configuration().setMessageVerbosity( coral::Debug );
    db.connect( connStr );
    db.transaction().start();
    boost::shared_ptr< int > r0 = db.fetch< int >( id0 );
    std::cout << "** read data contid="<<id0.containerId()<<" oid="<<id0.itemId()<<" data="<<*r0<<std::endl;
    boost::shared_ptr< std::string > r1 = db.fetch< std::string >( id1 );
    std::cout << "** read data contid="<<id1.containerId()<<" oid="<<id1.itemId()<<" data="<<*r1<<std::endl;
    db.transaction().commit();
  }catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  ::sleep(1);
  try{
    ora::Database db;
    db.configuration().setMessageVerbosity( coral::Debug );
    db.connect( connStr );
    db.transaction().start( false );
    if(db.exists()) db.drop();
    db.transaction().commit();
    db.disconnect();
  } catch ( const ora::Exception& e ){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }catch ( const std::exception& e){
    std::cout << "# Test Error: "<<e.what()<<std::endl;
  }
  
}


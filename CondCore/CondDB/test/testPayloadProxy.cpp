#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/PayloadProxy.h"
//
#include "MyTestData.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

int main (int argc, char** argv)
{
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string connectionString("sqlite_file:cms_conditions_2.db");
  std::cout <<"# Connecting with db in "<<connectionString<<std::endl;
  try{

    //*************
    ConnectionPool connPool;
    connPool.setMessageVerbosity( coral::Debug );
    Session session = connPool.createSession( connectionString, true );
    session.transaction().start( false );
    MyTestData d0( 20000 );
    MyTestData d1( 30000 );
    std::cout <<"# Storing payloads..."<<std::endl;
    cond::Hash p0 = session.storePayload( d0, boost::posix_time::microsec_clock::universal_time() );
    cond::Hash p1 = session.storePayload( d1, boost::posix_time::microsec_clock::universal_time() );
    std::string d2("abcd1234");
    cond::Hash p2 = session.storePayload( d2, boost::posix_time::microsec_clock::universal_time() );
    std::string d3("abcd1234");
    cond::Hash p3 = session.storePayload( d3, boost::posix_time::microsec_clock::universal_time() );
    IOVEditor editor;
    if( !session.existsIov( "MyNewIOV2" ) ){
      editor = session.createIov<MyTestData>( "MyNewIOV2", cond::runnumber ); 
      editor.setDescription("Test with MyTestData class");
      editor.insert( 1, p0 );
      editor.insert( 100, p1 );
      std::cout <<"# inserted 2 iovs..."<<std::endl;
      editor.flush();
      std::cout <<"# iov changes flushed..."<<std::endl;
    }
    if( !session.existsIov( "StringData2" ) ){
      editor = session.createIov<std::string>( "StringData2", cond::timestamp );
      editor.setDescription("Test with std::string class");
      editor.insert( 1000000, p2 );
      editor.insert( 2000000, p3 );
      editor.flush();
    }
    if( !session.existsIov( "StringData3" ) ){
      editor = session.createIov<std::string>( "StringData3", cond::lumiid );
      editor.setDescription("Test with std::string class");
      editor.insert( 4294967297, p2 );
      editor.flush();
    }

    session.transaction().commit();
    std::cout <<"# iov changes committed!..."<<std::endl;
    ::sleep(2);

    PayloadProxy<MyTestData> pp0;
    pp0.setUp( session );
    PayloadProxy<std::string> pp1;
    pp1.setUp( session );

    pp0.loadTag( "MyNewIOV2" );
    cond::ValidityInterval v1 = pp0.setIntervalFor( 25, true );
    const MyTestData& rd0 = pp0();
    if( rd0 != d0 ){
      std::cout <<"ERROR: MyTestData object read different from source."<<std::endl;
    } else {
      std::cout << "MyTestData instance valid from "<< v1.first<<" to "<<v1.second<<std::endl; 
    }
    cond::ValidityInterval v2 = pp0.setIntervalFor( 35, true );
    const MyTestData& rd1 = pp0();
    if( rd1 != d0 ){
      std::cout <<"ERROR: MyTestData object read different from source."<<std::endl;
    } else {
      std::cout << "MyTestData instance valid from "<< v2.first<<" to "<<v2.second<<std::endl; 
    }
    cond::ValidityInterval v3 = pp0.setIntervalFor( 100000, true );
    const MyTestData& rd2 = pp0();
    if( rd2 != d1 ){
      std::cout <<"ERROR: MyTestData object read different from source."<<std::endl;
    } else {
      std::cout << "MyTestData instance valid from "<< v3.first<<" to "<<v3.second<<std::endl; 
    }

    pp1.loadTag( "StringData2" );
    try{
      pp1.setIntervalFor( 345 );
    } catch ( cond::persistency::Exception& e ){
      std::cout <<"Expected error: "<<e.what()<<std::endl;
    }
    cond::ValidityInterval vs1 = pp1.setIntervalFor( 1000000, true );
    const std::string& rd3 = pp1();
    if( rd3 != d2 ){
      std::cout <<"ERROR: std::string object read different from source."<<std::endl;
    } else {
      std::cout << "std::string instance valid from "<< vs1.first<<" to "<<vs1.second<<std::endl; 
    }
    cond::ValidityInterval vs2 = pp1.setIntervalFor( 3000000, true );
    const std::string& rd4 = pp1();
    if( rd4 != d3 ){
      std::cout <<"ERROR: std::string object read different from source."<<std::endl;
    } else {
      std::cout << "std::string instance valid from "<< vs2.first<<" to "<<vs2.second<<std::endl; 
    }

    PayloadProxy<std::string> pp2;
    pp2.setUp( session );
    pp2.loadTag( "StringData3" );
    try{
      pp2.setIntervalFor( 4294967296 );
    } catch ( cond::persistency::Exception& e ){
      std::cout <<"Expected error: "<<e.what()<<std::endl;
    }
    
  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}


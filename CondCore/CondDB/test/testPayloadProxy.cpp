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

    IOVEditor editor = session.createIov<MyTestData>( "MyNewIOV2", cond::runnumber ); 
    editor.setDescription("Test with MyTestData class");
    editor.insert( 1, p0 );
    editor.insert( 100, p1 );
    std::cout <<"# inserted 2 iovs..."<<std::endl;
    editor.flush();
    std::cout <<"# iov changes flushed..."<<std::endl;

    editor = session.createIov<std::string>( "StringData2", cond::timestamp );
    editor.setDescription("Test with std::string class");
    editor.insert( 1000000, p2 );
    editor.insert( 2000000, p3 );
    editor.flush();

    session.transaction().commit();
    std::cout <<"# iov changes committed!..."<<std::endl;
    ::sleep(2);

    PayloadProxy<MyTestData> pp0( session );
    PayloadProxy<std::string> pp1( session );

    pp0.loadTag( "MyNewIOV2" );
    cond::ValidityInterval v1 = pp0.setIntervalFor( 25 );
    const MyTestData& rd0 = pp0();
    if( rd0 != d0 ){
      std::cout <<"ERROR: MyTestData object read different from source."<<std::endl;
    } else {
      std::cout << "MyTestData instance valid from "<< v1.first<<" to "<<v1.second<<std::endl; 
    }
    cond::ValidityInterval v2 = pp0.setIntervalFor( 35 );
    const MyTestData& rd1 = pp0();
    if( rd1 != d0 ){
      std::cout <<"ERROR: MyTestData object read different from source."<<std::endl;
    } else {
      std::cout << "MyTestData instance valid from "<< v2.first<<" to "<<v2.second<<std::endl; 
    }
    cond::ValidityInterval v3 = pp0.setIntervalFor( 100000 );
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
    cond::ValidityInterval vs1 = pp1.setIntervalFor( 1000000 );
    const std::string& rd3 = pp1();
    if( rd3 != d2 ){
      std::cout <<"ERROR: std::string object read different from source."<<std::endl;
    } else {
      std::cout << "std::string instance valid from "<< vs1.first<<" to "<<vs1.second<<std::endl; 
    }
    cond::ValidityInterval vs2 = pp1.setIntervalFor( 3000000 );
    const std::string& rd4 = pp1();
    if( rd4 != d3 ){
      std::cout <<"ERROR: std::string object read different from source."<<std::endl;
    } else {
      std::cout << "std::string instance valid from "<< vs2.first<<" to "<<vs2.second<<std::endl; 
    }
    
  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}


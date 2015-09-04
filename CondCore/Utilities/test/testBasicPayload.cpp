#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/Common/interface/BasicPayload.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

int run( const std::string& connectionString ){
  try{

    //*************
    std::cout <<"> Connecting with db in "<<connectionString<<std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity( coral::Debug );
    Session session = connPool.createSession( connectionString, true, cond::COND_DB );
    session.transaction().start( false );
    IOVEditor editor;
    if( !session.existsDatabase() || !session.existsIov( "BasicPayload_v0" ) ){
      editor = session.createIov<cond::BasicPayload>( "BasicPayload_v0", cond::runnumber ); 
      editor.setDescription("Test for timestamp selection");
    }
    for( int i=0;i<10;i++ ){
      cond::BasicPayload  p( i*10.1, i+1. );
      auto pid = session.storePayload( p );
      editor.insert( i*100+1, pid );
    }
    editor.flush();
    std::cout <<"> iov changes flushed..."<<std::endl;
    session.transaction().commit();
  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
  std::cout <<"## Run successfully completed."<<std::endl;
  return 0;
}

int main (int argc, char** argv)
{
  int ret = 0;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::string connectionString0("sqlite_file:BasicPayload_v0.db");
  ret = run( connectionString0 );
  return ret;
}


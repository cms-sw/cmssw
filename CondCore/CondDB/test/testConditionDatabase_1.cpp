#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

void readTag( const std::string& tag, Session& session, const boost::posix_time::ptime& snapshotTime ){
  IOVProxy proxy;
  if( snapshotTime.is_not_a_date_time() ) proxy = session.readIov( tag );
  else proxy = session.readIov( tag, snapshotTime );
  std::cout <<"> iov loaded size="<<proxy.loadedSize()<<std::endl;
  std::cout <<"> iov sequence size="<<proxy.sequenceSize()<<std::endl;
  IOVProxy::Iterator iovIt = proxy.find( 107 );
  if( iovIt == proxy.end() ){
    std::cout <<">[0] not found!"<<std::endl;
  } else {
    cond::Iov_t val = *iovIt;
    std::cout <<"#[0] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
    boost::shared_ptr<std::string> pay0 = session.fetchPayload<std::string>( val.payloadId );
    std::cout <<"#[0] payload="<<*pay0<<std::endl;
  }
  iovIt = proxy.find( 235 );
  if( iovIt == proxy.end() ){
    std::cout <<">[1] not found!"<<std::endl;
  } else {
    cond::Iov_t val = *iovIt;
    std::cout <<"#[1] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
    boost::shared_ptr<std::string> pay0 = session.fetchPayload<std::string>( val.payloadId );
    std::cout <<"#[1] payload="<<*pay0<<std::endl;
  }
}

int run( const std::string& connectionString ){
  try{

    //*************
    std::cout <<"> Connecting with db in "<<connectionString<<std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity( coral::Debug );
    Session session = connPool.createSession( connectionString, true, cond::COND_DB );
    session.transaction().start( false );
    std::string pay0("Payload #0");
    std::string pay1("Payload #1");
    std::string pay2("Payload #2");
    std::string pay3("Payload #3");
    std::string pay4("Payload #4");
    std::string pay5("Payload #5");
    auto p0 = session.storePayload( pay0 );
    auto p1 = session.storePayload( pay1 );
    auto p2 = session.storePayload( pay2 );

    IOVEditor editor;
    if( !session.existsIov( "MyTag" ) ){
      editor = session.createIov<std::string>( "MyTag", cond::runnumber ); 
      editor.setDescription("Test for timestamp selection");
      editor.insert( 1, p0 );
      editor.insert( 101, p1 );
      editor.insert( 201, p2 );
      std::cout <<"> inserted 3 iovs..."<<std::endl;
      editor.flush();
      std::cout <<"> iov changes flushed..."<<std::endl;
    }
    session.transaction().commit();
    boost::posix_time::ptime snap0 = boost::posix_time::microsec_clock::universal_time();
    std::cout <<"> iov changes committed!..."<<std::endl;
    ::sleep(2);
    boost::posix_time::ptime notime;
    session.transaction().start();
    readTag( "MyTag", session, notime );
    session.transaction().commit();
    session.transaction().start( false );
    auto p3 = session.storePayload( pay3 );
    auto p4 = session.storePayload( pay4 );
    auto p5 = session.storePayload( pay5 );
    editor = session.editIov( "MyTag" );
    editor.insert( 101, p3 );
    editor.insert( 222, p4 );
    editor.flush();
    session.transaction().commit();
    boost::posix_time::ptime snap1 = boost::posix_time::microsec_clock::universal_time();
    ::sleep(2);
    session.transaction().start();
    readTag( "MyTag", session, notime );
    session.transaction().commit();
    session.transaction().start( false );
    editor = session.editIov( "MyTag" );
    editor.insert( 102, p5 );
    editor.flush();
    session.transaction().commit();
    session.transaction().start();
    readTag( "MyTag", session, notime );
    session.transaction().commit();
    session.transaction().start();
    readTag( "MyTag", session, snap0 );
    session.transaction().commit();
    session.transaction().start();
    readTag( "MyTag", session, snap1 );
    session.transaction().commit();
    // 
    session.transaction().start( false );
    GTEditor gtWriter = session.createGlobalTag("MY_TEST_GT_V0");
    gtWriter.setDescription( "test GT" );
    gtWriter.setRelease( "CMSSW_7_5_X" );
    gtWriter.setSnapshotTime( snap0 );
    gtWriter.insert( "myrecord", "MyTag" );
    gtWriter.flush();
    session.transaction().commit();
    session.transaction().start();
    GTProxy gtReader = session.readGlobalTag("MY_TEST_GT_V0");
    boost::posix_time::ptime snap2 = gtReader.snapshotTime();
    readTag( "MyTag", session, snap2 );
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
  std::string connectionString0("sqlite_file:cms_conditions_1.db");
  ret = run( connectionString0 );
  return ret;
}


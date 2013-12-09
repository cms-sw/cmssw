#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/CondDB.h"
//
#include "MyTestData.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::db;

int main (int argc, char** argv)
{
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  //std::string connectionString("oracle://cms_orcoff_prep/CMS_CONDITIONS");
  //std::string connectionString("sqlite_file:/build/gg/cms_conditions.db");
  //std::string connectionString("sqlite_file:cms_conditions.db");
  std::string connectionString("sqlite_file:cms_conditions_ora.db");
  std::cout <<"# Connecting with db in "<<connectionString<<std::endl;
  try{

    //*************
    Session session;
    session.configuration().setMessageVerbosity( coral::Debug );
    session.open( connectionString );
    session.transaction().start( false );
    MyTestData d0( 17 );
    MyTestData d1( 999 );
  std::cout <<"# Storing payloads..."<<std::endl;
    cond::Hash p0 = session.storePayload( d0, boost::posix_time::microsec_clock::universal_time() );
    cond::Hash p1 = session.storePayload( d1, boost::posix_time::microsec_clock::universal_time() );
    std::string d("abcd1234");
    cond::Hash p3 = session.storePayload( d, boost::posix_time::microsec_clock::universal_time() );

    IOVEditor editor = session.createIov<MyTestData>( "MyNewIOV", cond::runnumber ); 
    editor.setDescription("Test with MyTestData class");
    editor.insert( 1, p0 );
    editor.insert( 100, p1 );
    std::cout <<"# inserted 2 iovs..."<<std::endl;
    editor.flush();
    std::cout <<"# iov changes flushed..."<<std::endl;

    editor = session.createIov<std::string>( "StringData", cond::timestamp );
    editor.setDescription("Test with std::string class");
    editor.insert( 1000000, p3 );
    editor.insert( 2000000, p3 );
    editor.flush();

    session.transaction().commit();
    std::cout <<"# iov changes committed!..."<<std::endl;
    ::sleep(2);
    session.transaction().start();

    IOVProxy proxy = session.readIov( "MyNewIOV" );
    IOVProxy::Iterator iovIt = proxy.find( 57 );
    if( iovIt == proxy.end() ){
      std::cout <<"#0 not found!"<<std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout <<"#0 iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay0 = session.fetchPayload<MyTestData>( val.payloadId );
      pay0->print();
    }
    iovIt++;
    if(iovIt == proxy.end() ){
      std::cout<<"#1 not found!"<<std::endl;
    } else {
      cond::Iov_t val =*iovIt;
      std::cout <<"#1 iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay1 = session.fetchPayload<MyTestData>( val.payloadId );
      pay1->print();
    }
    iovIt = proxy.find( 176 );
    if( iovIt == proxy.end() ){
      std::cout <<"#2 not found!"<<std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout <<"#2 iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay2 = session.fetchPayload<MyTestData>( val.payloadId );
      pay2->print();
    }
    iovIt++;
    if(iovIt == proxy.end() ){
      std::cout<<"#3 not found!"<<std::endl;
    } else {
      cond::Iov_t val =*iovIt;
      std::cout <<"#3 iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay3 = session.fetchPayload<MyTestData>( val.payloadId );
      pay3->print();
    }

    proxy = session.readIov( "StringData" ); 
    auto iov2It = proxy.find( 1000022 );
    if(iov2It == proxy.end() ){
      std::cout<<"#4 not found!"<<std::endl;
    } else {
      cond::Iov_t val =*iov2It;
      std::cout <<"#4 iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<std::string> pay4 = session.fetchPayload<std::string>( val.payloadId );
      std::cout <<" ## pay4="<<*pay4<<std::endl;
    }
    session.transaction().commit();
  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}


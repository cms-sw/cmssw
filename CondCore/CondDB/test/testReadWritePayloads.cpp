
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/ConnectionPool.h"
//
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "MyTestData.h"
// #include "ArrayPayload.h"
// #include "SimplePayload.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace cond::persistency;

// define values here, so we can check them after reading back ...
const int iVal0( 18); // 17
const int iVal1(909); // 999
const std::string sVal("7890abcd"); // "abcd1234"


int doWrite( const std::string& connectionString ){
  try{

    //*************
    std::cout <<"> Connecting for writing with db in "<<connectionString<<std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity( coral::Debug );
    Session session = connPool.createSession( connectionString, true, cond::COND_DB );
    session.transaction().start( false );

    MyTestData d0( iVal0 );
    MyTestData d1( iVal1 );
    std::cout <<"> Storing payload ptr=" << &d0 << std::endl;
    cond::Hash p0 = session.storePayload( d0, boost::posix_time::microsec_clock::universal_time() );
    cond::Hash p1 = session.storePayload( d1, boost::posix_time::microsec_clock::universal_time() );

    std::string d(sVal);
    cond::Hash p3 = session.storePayload( d, boost::posix_time::microsec_clock::universal_time() );

    // ArrayPayload arrayPl( );
    // cond::Hash ap0 = session.storePayload( arrayPl, boost::posix_time::microsec_clock::universal_time() );

    // SimplePayload simplePl( 43 );
    // cond::Hash sp0 = session.storePayload( simplePl, boost::posix_time::microsec_clock::universal_time() );

    IOVEditor editor;
    if( !session.existsIov( "MyNewIOV" ) ){
      editor = session.createIov<MyTestData>( "MyNewIOV", cond::runnumber ); 
      editor.setDescription("Test with MyTestData class");
      editor.insert( 1, p0 );
      editor.insert( 100, p1 );
      std::cout <<"> inserted 2 iovs..."<<std::endl;
      editor.flush();
      std::cout <<"> iov changes flushed..."<<std::endl;
    }

    if( !session.existsIov( "StringData" ) ){
      editor = session.createIov<std::string>( "StringData", cond::timestamp );
      editor.setDescription("Test with std::string class");
      editor.insert( 1000000, p3 );
      editor.insert( 2000000, p3 );
      editor.flush();
    }

//     if( !session.existsIov( "ArrayData" ) ){
//       editor = session.createIov<ArrayPayload>( "ArrayData", cond::timestamp );
//       editor.setDescription("Test with ArrayPayload class");
//       editor.insert( 1000000, ap0 );
//       editor.insert( 2000000, ap0 );
//       editor.flush();
//     }
// 
//     if( !session.existsIov( "SimpleData" ) ){
//       editor = session.createIov<SimplePayload>( "SimpleData", cond::timestamp );
//       editor.setDescription("Test with SimplePayload class");
//       editor.insert( 1000000, sp0 );
//       editor.insert( 2000000, sp0 );
//       editor.flush();
//     }

    session.transaction().commit();
    std::cout <<"> iov changes committed!..."<<std::endl;

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

int doRead( const std::string& connectionString ){

  int nFail = 0;
  try{

    //*************
    std::cout <<"> Connecting for reading with db in "<<connectionString<<std::endl;
    ConnectionPool connPool;
    connPool.setMessageVerbosity( coral::Debug );
    Session session = connPool.createSession( connectionString, true, cond::COND_DB );
    session.transaction().start( false );

    session.transaction().start();

    IOVProxy proxy = session.readIov( "MyNewIOV" );
    std::cout <<"> iov loaded size="<<proxy.loadedSize()<<std::endl;
    std::cout <<"> iov sequence size="<<proxy.sequenceSize()<<std::endl;
    IOVProxy::Iterator iovIt = proxy.find( 57 );
    if( iovIt == proxy.end() ){
      std::cout <<">[0] not found!"<<std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout <<"#[0] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay0 = session.fetchPayload<MyTestData>( val.payloadId );
      pay0->print();
      if ( *pay0 != MyTestData(iVal0) ){
	nFail++;
	std::cout << "ERROR, pay0 found to be wrong, expected : " << iVal0 << " IOV: " << val.since << std::endl;
      }
      iovIt++;
    }
    if(iovIt == proxy.end() ){
      std::cout<<"#[1] not found!"<<std::endl;
    } else {
      cond::Iov_t val =*iovIt;
      std::cout <<"#[1] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay1 = session.fetchPayload<MyTestData>( val.payloadId );
      pay1->print();
      if ( *pay1 != MyTestData(iVal1) ){
	nFail++;
	std::cout << "ERROR, pay1 found to be wrong, expected : " << iVal1 << " IOV: " << val.since << std::endl;
      }
    }
    iovIt = proxy.find( 176 );
    if( iovIt == proxy.end() ){
      std::cout <<"#[2] not found!"<<std::endl;
    } else {
      cond::Iov_t val = *iovIt;
      std::cout <<"#[2] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay2 = session.fetchPayload<MyTestData>( val.payloadId );
      pay2->print();
      if ( *pay2 != MyTestData(iVal1) ){
	nFail++;
	std::cout << "ERROR, pay2 found to be wrong, expected : " << iVal1 << " IOV: " << val.since << std::endl;
      }
      iovIt++;
    }
    if(iovIt == proxy.end() ){
      std::cout<<"#[3] not found!"<<std::endl;
    } else {
      cond::Iov_t val =*iovIt;
      std::cout <<"#[3] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<MyTestData> pay3 = session.fetchPayload<MyTestData>( val.payloadId );
      pay3->print();
      if ( *pay3 != MyTestData(iVal1) ){
	nFail++;
	std::cout << "ERROR, pay3 found to be wrong, expected : " << iVal1 << " IOV: " << val.since << std::endl;
      }
    }

    proxy = session.readIov( "StringData" ); 
    auto iov2It = proxy.find( 1000022 );
    if(iov2It == proxy.end() ){
      std::cout<<"#[4] not found!"<<std::endl;
    } else {
      cond::Iov_t val =*iov2It;
      std::cout <<"#[4] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
      boost::shared_ptr<std::string> pay4 = session.fetchPayload<std::string>( val.payloadId );
      std::cout <<"#pay4="<<*pay4<<std::endl;
      if ( *pay4 != sVal ){
	nFail++;
	std::cout << "ERROR, pay4 found to be " << *pay4 << " expected : " << sVal << " IOV: " << val.since << std::endl;
      }
    }

//     proxy = session.readIov( "ArrayData" ); 
//     auto iov3It = proxy.find( 1000022 );
//     if(iov3It == proxy.end() ){
//       std::cout<<"#[5] not found!"<<std::endl;
//     } else {
//       cond::Iov_t val =*iov3It;
//       std::cout <<"#[5] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
//       boost::shared_ptr<ArrayPayload> pay5 = session.fetchPayload<ArrayPayload>( val.payloadId );
//       // std::cout << "#pay5=" << (*pay5==arrayPl ? "OK" : "NOT OK") << std::endl;
//     }
// 
//     proxy = session.readIov( "SimpleData" ); 
//     auto iov4It = proxy.find( 1000022 );
//     if(iov4It == proxy.end() ){
//       std::cout<<"#[6] not found!"<<std::endl;
//     } else {
//       cond::Iov_t val =*iov4It;
//       std::cout <<"#[6] iov since="<<val.since<<" till="<<val.till<<" pid="<<val.payloadId<<std::endl;
//       boost::shared_ptr<SimplePayload> pay5 = session.fetchPayload<SimplePayload>( val.payloadId );
//       // std::cout << "#pay6=" << (*pay6==simplePl ? "OK" : "NOT OK") << std::endl;
//     }


    session.transaction().commit();
  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
  if (nFail == 0) {
    std::cout << "## Run successfully completed." << std::endl;
  } else {
    std::cout << "## Run completed with ERRORS. nFail = " << nFail << std::endl;
  }

  return nFail;
}

int run( const std::string& connectionString ){

  int ret = doWrite( connectionString ); 
  if ( ret != 0 ) return ret;

  ::sleep(2);
  ret = doRead ( connectionString );

  return ret;
}


int main (int argc, char** argv)
{
  int ret = 0;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  std::cout <<"## Running with CondDBV2 format..."<<std::endl;

  if (argc < 3) {
    std::cout << "Not enough arguments given, assuming automatic run in unit-tests. Will not do anything. " << std::endl;
    std::cout << "If you did not expect this, please run it with the arguments as follows:" << std::endl;
    std::cout << "testReadWritePayloads (write|read) <dbName> " << std::endl;
    return 0;
  }

  std::string connectionString0("sqlite_file:cms_conditions_v2.db");
  if ( std::string(argv[2]).size() > 3 ) {
    connectionString0 = std::string( argv[2] );
  }


  if (std::string(argv[1]) == "write") {
    ret = doWrite( connectionString0 );
  } else if (std::string(argv[1]) == "read") {
    ret = doRead ( connectionString0 );
  } else {
    ret = run( connectionString0 );
  }

  return ret;
}


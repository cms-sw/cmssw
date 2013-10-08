#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
//
#include "CondCore/CondDB/interface/CondDB.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
//
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>

void dumpSince( conddb::Time_t target, conddb::IOVProxy& data ){
  auto i = data.find( target );
  if( i == data.end() ) {
    std::cout <<"No valid IOV for time="<<target<<std::endl;
  } else std::cout <<"IOV for time="<<target<<" has a since="<<(*i).since<<std::endl;
}

int main (int argc, char** argv)
{
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::string connectionString0("sqlite_file:cond_ora.db");
  std::string connectionString1("sqlite_file:cond_new.db");
  std::string tag0("my_tag_0");
  std::string tag1("my_tag_1");
  std::string tag2("my_tag_2");
  try{

    // prepare an old-fashion db...
    cond::DbConnection oraConn;
    cond::DbSession oraSession = oraConn.createSession();
    oraSession.open( connectionString0 );
    cond::IOVEditor oraEditor( oraSession );
    cond::MetaData metadata( oraSession );
    oraSession.transaction().start( false );
    oraEditor.createIOVContainerIfNecessary();
    std::string iovTok = oraEditor.create( cond::runnumber, 999999, "metadata xyz" );
    std::string data0("X=1;Y=6;Z=11");
    std::string data1("X=2;Y=7;Z=12");
    std::string data2("X=3;Y=8;Z=13");
    std::string data3("X=4;Y=9;Z=14");
    std::string data4("X=5;Y=10;Z=15");
    std::string tok0 = oraSession.storeObject(&data0,"MyData"); 
    std::string tok1 = oraSession.storeObject(&data1,"MyData"); 
    std::string tok2 = oraSession.storeObject(&data2,"MyData"); 
    std::string tok3 = oraSession.storeObject(&data3,"MyData"); 
    std::string tok4 = oraSession.storeObject(&data4,"MyData"); 
    oraEditor.append( 1000, tok0 );
    oraEditor.append( 2000, tok1 );
    oraEditor.append( 3000, tok2 );
    oraEditor.append( 4000, tok3 );
    oraEditor.append( 5000, tok4 );
    metadata.addMapping( tag0, iovTok, cond::runnumber ); 
    oraSession.transaction().commit();

    // read the old db with the wrapper:
    conddb::Session session;
    conddb::IOVProxy reader;
    conddb::IOVEditor editor;
    session.open( connectionString0, true );
    session.transaction().start( true );
    std::cout <<"Database "<<connectionString0<<" does ";
    if( !session.existsDatabase() ) {
      std::cout <<"not ";
    }
    std::cout<<"exist."<<std::endl;
    reader = session.readIov( tag0 );
    std::cout <<"Size="<<reader.size()<<std::endl;
    dumpSince( 99, reader );
    dumpSince( 3650, reader );
    for( auto iov: reader ){
      std::cout <<"#Since "<<iov.since<<" Till "<<iov.till<<" PID "<<iov.payloadId<<std::endl;      
    }
    session.transaction().commit();
    session.close();
    session.open( connectionString0 );
    session.transaction().start( false );
    editor = session.editIov( tag0 );
    std::string data5("X=6;Y=11;Z=16");
    std::string data6("X=7;Y=12;Z=17");
    std::string data7("X=8;Y=13;Z=18");
    std::string data8("X=9;Y=14;Z=19");
    std::string data9("X=10;Y=15;Z=20");
    boost::posix_time::ptime t = boost::posix_time::microsec_clock::universal_time();
    conddb::Hash h0 = session.storePayload(data0, t ); 
    conddb::Hash h1 = session.storePayload(data1, t ); 
    conddb::Hash h2 = session.storePayload(data2, t ); 
    conddb::Hash h3 = session.storePayload(data3, t ); 
    conddb::Hash h4 = session.storePayload(data4, t ); 
    editor.insert( 6000, h0 );
    editor.insert( 7000, h1 );
    editor.insert( 8000, h2 );
    editor.insert( 9000, h3 );
    editor.insert( 10000, h0 );
    editor.setLastValidatedTime( 10000 );
    editor.flush();
    session.transaction().commit();
    session.transaction().start( true );
    reader = session.readIov( tag0 );
    std::cout <<"Size="<<reader.size()<<std::endl;
    for( auto iov: reader ){
      std::cout <<"#Since "<<iov.since<<" Till "<<iov.till<<" PID "<<iov.payloadId<<std::endl;      
    }
    session.transaction().commit();
    session.close();

    //********
    session.open(connectionString1);
    session.transaction().start( false);
    std::cout <<"Database "<<connectionString1<<" does ";
    if( !session.existsDatabase() ) {
      std::cout <<"not ";
    }
    std::cout<<"exist."<<std::endl;
    try{
      editor = session.editIov( tag1 );
    } catch ( conddb::Exception& e ){
      std::cout <<"ERROR: "<<e.what()<<std::endl;
    }
    editor = session.createIov<std::string>( tag1, conddb::time::RUNNUMBER, conddb::OFFLINE );
    std::cout <<"Now the database "<<connectionString1<<" does ";
    if( !session.existsDatabase() ) {
      std::cout <<"not ";
    }
    std::cout<<"exist."<<std::endl;
    editor.setDescription("My stuff...");
    editor.setEndOfValidity( 123456 );
    std::string d0("Bla bla bla 0");
    std::string d1("Bla bla bla 1");
    std::string d2("Bla bla bla 2");
    conddb::Hash p0 = session.storePayload(d0, boost::posix_time::microsec_clock::universal_time());
    conddb::Hash p1 = session.storePayload(d1, boost::posix_time::microsec_clock::universal_time());
    conddb::Hash p2 = session.storePayload(d2, boost::posix_time::microsec_clock::universal_time());
    editor.insert( 100, p0 );
    editor.insert( 200, p1 );
    editor.insert( 300, p2 );
    editor.setLastValidatedTime( 300 );
    editor.flush();
    session.transaction().commit();

    std::cout<<"Reading back..."<<std::endl;
    session.transaction().start( true );
    reader = session.readIov( tag1 );
    std::cout <<"Tag "<<reader.tag()<<" timeType:"<<conddb::time::timeTypeName(reader.timeType())<<" size:"<<reader.size()<<
      " type:"<<reader.payloadObjectType()<<" endOfValidity:"<<reader.endOfValidity()<<" lastValidatedTime:"<<reader.lastValidatedTime()<<std::endl;
    reader.find( 12 );
    std::cout <<"Now size="<<reader.size()<<std::endl;
    conddb::Iov_t iov = reader.getInterval( 235 );
    std::cout <<"Since:"<<iov.since<<" till:"<<iov.till<<" pid:"<<iov.payloadId<<std::endl;

    for( auto iov: reader ){
      std::cout <<"#Since "<<iov.since<<" Till "<<iov.till<<" PID "<<iov.payloadId<<std::endl;
    }    
    session.transaction().commit();
    session.close();
  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}


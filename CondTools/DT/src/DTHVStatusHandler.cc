/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/25 11:00:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTHVStatusHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTDBSession.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/TimeStamp.h"

//---------------
// C++ Headers --
//---------------
#include <map>
#include <sys/time.h>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTHVStatusHandler::DTHVStatusHandler( const edm::ParameterSet& ps ) :
 dataTag(               ps.getParameter<std::string> ( "tag" ) ),
 onlineConnect(         ps.getParameter<std::string> ( "onlineDB" ) ),
 onlineAuthentication(  ps.getParameter<std::string> ( 
                        "onlineAuthentication" ) ),
 bufferConnect(         ps.getParameter<std::string> ( "bufferDB" ) ),
 ySince(                ps.getParameter<int> ( "sinceYear"  ) ),
 mSince(                ps.getParameter<int> ( "sinceMonth" ) ),
 dSince(                ps.getParameter<int> ( "sinceDay"   ) ),
 yUntil(                ps.getParameter<int> ( "untilYear"  ) ),
 mUntil(                ps.getParameter<int> ( "untilMonth" ) ),
 dUntil(                ps.getParameter<int> ( "untilDay"   ) ),
 mapVersion(            ps.getParameter<std::string> ( "mapVersion"   ) ) {
  std::cout << " PopCon application for DT HV data export "
            << onlineAuthentication
            << std::endl;
/*
*/
}

//--------------
// Destructor --
//--------------
DTHVStatusHandler::~DTHVStatusHandler() {
}

//--------------
// Operations --
//--------------
void DTHVStatusHandler::getNewObjects() {

  std::cout << "get new objects..." << std::endl;

// online DB connection

  cond::DBSession* omds_session;
  cond::Connection* omds_connect;
  cond::CoralTransaction* omds_transaction;

  std::cout << "open omds session... " << onlineAuthentication << std::endl;

  omds_session = new cond::DBSession();
  // to get the username/passwd from $CORAL_AUTH_PATH/authentication.xml
  omds_session->configuration().setAuthenticationMethod( cond::XML );
  omds_session->configuration().setAuthenticationPath( onlineAuthentication );
  // set message level to Error or Debug
  omds_session->configuration().setMessageLevel( cond::Error );
//  omds_session->connectionConfiguration().setConnectionRetrialTimeOut( 60 );
  omds_session->open();

  std::cout << "omds session open, start transaction" << std::endl;

  omds_connect = new cond::Connection( onlineConnect );
  omds_connect->connect( omds_session );
  omds_transaction = &( omds_connect->coralTransaction() );
  omds_transaction->start( true );

  std::cout << "omds transaction started" << std::endl;

  std::cout << "get omds session proxy... " << std::endl;
  omds_s_proxy = &( omds_transaction->coralSessionProxy() );
  std::cout << "omds session proxy got" << std::endl;
  omds_transaction->start( true );

// buffer DB connection

  cond::DBSession* buff_session;
  cond::Connection* buff_connect;
  cond::CoralTransaction* buff_transaction;

  std::cout << "open buffer session..." << std::endl;

  buff_session = new cond::DBSession();
  // to get the username/passwd from $CORAL_AUTH_PATH/authentication.xml
  buff_session->configuration().setAuthenticationMethod( cond::XML );
  buff_session->configuration().setAuthenticationPath( onlineAuthentication );
  // set message level to Error or Debug
  buff_session->configuration().setMessageLevel( cond::Error );
//  buff_session->connectionConfiguration().setConnectionRetrialTimeOut( 60 );
  buff_session->open();

  std::cout << "buffer session open, start transaction" << std::endl;

  buff_connect = new cond::Connection( bufferConnect );
  buff_connect->connect( buff_session );
  buff_transaction = &( buff_connect->coralTransaction() );
  buff_transaction->start( false );

  std::cout << "buffer transaction started" << std::endl;

  std::cout << "get buffer session proxy... " << std::endl;
  buff_s_proxy = &( buff_transaction->coralSessionProxy() );
  std::cout << "buffer session proxy got" << std::endl;
  buff_transaction->start( false );

// offline info

  //to access the information on the tag inside the offline database:
  cond::TagInfo const & ti = tagInfo();
//  unsigned int last = ti.lastInterval.first;
  cond::Time_t last = ti.lastInterval.first;
  std::cout << "latest DCS data (HV) already copied until: "
            << last << std::endl;

  if ( last == 0 ) {
    DTHVStatus* dummyConf = new DTHVStatus( dataTag );
    cond::Time_t snc = 1;
    m_to_transfer.push_back( std::make_pair( dummyConf, snc ) );
  }

  //to access the information on last successful log entry for this tag:
//  cond::LogDBEntry const & lde = logDBEntry();     

  //to access the lastest payload (Ref is a smart pointer)
  Ref payload = lastPayload();

//  unsigned lastIOV = last;
  cond::Time_t lastIOV = last;
  std::cout << "check for new data since " << lastIOV << std::endl;

  std::set<std::string> lt( omds_s_proxy->nominalSchema().listTables() );
  std::set<std::string>::const_iterator iter = lt.begin();
  std::set<std::string>::const_iterator iend = lt.end();
  while ( iter != iend ) {
    const std::string& istr = *iter++;
    std::cout << "TABLE: " << istr << std::endl;
  }

  getLayerSplit();
  getChannelMap();

  std::cout << "open buffer db..." << std::endl;
  if ( !( buff_s_proxy->nominalSchema().existsTable( "HVSNAPSHOT" ) ) ) {
    std::cout << "create snapshot description..." << std::endl;
    coral::TableDescription hvssDesc;
    hvssDesc.setName( "HVSNAPSHOT" );
    hvssDesc.insertColumn( "TIME",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(coral::TimeStamp) ) );
    hvssDesc.insertColumn( "WHEEL",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
    hvssDesc.insertColumn( "STATION",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
    hvssDesc.insertColumn( "SECTOR",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
    hvssDesc.insertColumn( "SUPERLAYER",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
    hvssDesc.insertColumn( "LAYER",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
    hvssDesc.insertColumn( "PART",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
//    hvssDesc.insertColumn( "VOLTAGE",
//                           coral::AttributeSpecification::typeNameForId( 
//                           typeid(float) ) );
//    hvssDesc.insertColumn( "CURRENT",
//                           coral::AttributeSpecification::typeNameForId( 
//                           typeid(float) ) );
    hvssDesc.insertColumn( "STATUS",
                           coral::AttributeSpecification::typeNameForId( 
                           typeid(int) ) );
    std::cout << "create snapshot table..." << std::endl;
    buff_s_proxy->nominalSchema().createTable( hvssDesc );
    updateHVStatus( 0 );
  }
  else {
    updateHVStatus( last );
  }

  delete omds_connect;
  delete omds_session;
  buff_transaction->commit();
  delete buff_connect;
  delete buff_session;
/*
*/
  return;

}


std::string DTHVStatusHandler::id() const {
  return "aaa";
//  return dataTag;
}


void DTHVStatusHandler::getChannelMap() {

  if ( !( buff_s_proxy->nominalSchema().existsTable( "HVALIASES" ) ) ) {
    dumpHVAliases();
  }
  else {
    std::cout << "retrieve aliases table..." << std::endl;
    coral::ITable& hvalTable =
      buff_s_proxy->nominalSchema().tableHandle( "HVALIASES" );
    std::auto_ptr<coral::IQuery>
      hvalQuery( hvalTable.newQuery() );
    hvalQuery->addToOutputList( "DETID" );
    hvalQuery->addToOutputList(  "DPID" );
    coral::ICursor& hvalCursor = hvalQuery->execute();
    int chId;
    int dpId;
    while ( hvalCursor.next() ) {
      chId = hvalCursor.currentRow()["DETID"].data<int>();
      dpId = hvalCursor.currentRow()[ "DPID"].data<int>();
      aliasMap.insert( std::pair<int,int>( dpId, chId ) );
    }
    std::map<int,int>::const_iterator aliasIter = aliasMap.begin();
    std::map<int,int>::const_iterator aliasIend = aliasMap.end();
    while ( aliasIter != aliasIend ) {
      const std::pair<int,int>& ientry = *aliasIter++;
      int dpId = ientry.first;
      int chId = ientry.second;
      DTWireId wireId( chId );
      std::cout << wireId.wheel     () << " "
                << wireId.station   () << " "
                << wireId.sector    () << " "
                << wireId.superLayer() << " "
                << wireId.layer     () << " "
                << wireId.wire      () - 10 << " "
                << dpId << std::endl;
    }
  }

  return;

}


void DTHVStatusHandler::getLayerSplit() {
  std::cout << "retrieve layer split table..." << std::endl;
  int whe;
  int sec;
  int sta;
  int qua;
  int lay;
  int l_p;
  int f_c;
  int l_c;
  coral::ITable& lsplTable =
    omds_s_proxy->nominalSchema().tableHandle( "DT_HV_LAYER_SPLIT" );
  std::auto_ptr<coral::IQuery>
    lsplQuery( lsplTable.newQuery() );
  coral::AttributeList versionBindVariableList;
  versionBindVariableList.extend( "version", typeid(std::string) );
  versionBindVariableList["version"].data<std::string>() = mapVersion;
  lsplQuery->setCondition( "VERSION=:version", versionBindVariableList );
  lsplQuery->addToOutputList( "WHEEL" );
  lsplQuery->addToOutputList( "SECTOR" );
  lsplQuery->addToOutputList( "STATION" );
  lsplQuery->addToOutputList( "SUPERLAYER" );
  lsplQuery->addToOutputList( "LAYER" );
  lsplQuery->addToOutputList( "PART" );
  lsplQuery->addToOutputList( "FIRST_CELL" );
  lsplQuery->addToOutputList( "LAST_CELL" );
  coral::ICursor& lsplCursor = lsplQuery->execute();
  while ( lsplCursor.next() ) {
    whe = lsplCursor.currentRow()["WHEEL"     ].data<int>();
    sec = lsplCursor.currentRow()["SECTOR"    ].data<int>();
    sta = lsplCursor.currentRow()["STATION"   ].data<int>();
    qua = lsplCursor.currentRow()["SUPERLAYER"].data<int>();
    lay = lsplCursor.currentRow()["LAYER"     ].data<int>();
    l_p = lsplCursor.currentRow()["PART"      ].data<int>();
    f_c = lsplCursor.currentRow()["FIRST_CELL"].data<int>();
    l_c = lsplCursor.currentRow()[ "LAST_CELL"].data<int>();
    DTWireId wireId( whe, sta, sec, qua, lay, 10 + l_p );
    laySplit.insert( std::pair<int,int>( wireId.rawId(), 
                                         ( f_c * 10000 ) + l_c ) );
  }
  std::map<int,int>::const_iterator iter = laySplit.begin();
  std::map<int,int>::const_iterator iend = laySplit.end();
  while ( iter != iend ) {
    const std::pair<int,int>& entry = *iter++;
    int rawId = entry.first;
    int split = entry.second;
    DTWireId id( rawId );
    int fc = split / 10000;
    int lc = split % 10000;
    std::cout << id.wheel()      << " "
              << id.sector()     << " "
              << id.station()    << " "
              << id.superLayer() << " "
              << id.layer()      << " "
              << id.wire()       << " "
              << fc << " " << lc << std::endl;
  }
}


void DTHVStatusHandler::dumpHVAliases() {

  std::cout << "DTHVStatusHandler::dumpHVAliases - begin" << std::endl;

  std::cout << "create aliases description..." << std::endl;
  coral::TableDescription hvalDesc;
  hvalDesc.setName( "HVALIASES" );
  hvalDesc.insertColumn( "DETID",
                         coral::AttributeSpecification::typeNameForId( 
                         typeid(int) ) );
  hvalDesc.insertColumn(  "DPID",
                         coral::AttributeSpecification::typeNameForId( 
                         typeid(int) ) );
  std::cout << "create aliases table..." << std::endl;
  coral::ITable& hvalTable = 
  buff_s_proxy->nominalSchema().createTable( hvalDesc );

  std::map<int,std::string> idMap;
  coral::ITable& dpidTable =
    omds_s_proxy->nominalSchema().tableHandle( "DP_NAME2ID" );
//  std::cout << "1" << std::endl;
  std::auto_ptr<coral::IQuery>
    dpidQuery( dpidTable.newQuery() );
//  std::cout << "2" << std::endl;
  dpidQuery->addToOutputList( "ID" );
//  std::cout << "3" << std::endl;
  dpidQuery->addToOutputList( "DPNAME" );
//  std::cout << "4" << std::endl;
  coral::ICursor& dpidCursor = dpidQuery->execute();
//  std::cout << "5" << std::endl;
  while( dpidCursor.next() ) {
//    std::cout << "6" << std::endl;
    const coral::AttributeList& row = dpidCursor.currentRow();
//    std::cout << "7" << std::endl;
    int id         = static_cast<int>( 
                     row["ID"    ].data<float>() );
//    std::cout << "8" << std::endl;
    std::string dp = row["DPNAME"].data<std::string>();
//    std::cout << "9" << std::endl;
    idMap.insert( std::pair<int,std::string>( id, dp ) );
  }

  std::map<std::string,std::string> cnMap;
  coral::ITable& nameTable =
    omds_s_proxy->nominalSchema().tableHandle( "ALIASES" );
  std::auto_ptr<coral::IQuery>
    nameQuery( nameTable.newQuery() );
  nameQuery->addToOutputList( "DPE_NAME" );
  nameQuery->addToOutputList( "ALIAS" );
  coral::ICursor& nameCursor = nameQuery->execute();
  while( nameCursor.next() ) {
    const coral::AttributeList& row = nameCursor.currentRow();
    std::string dp = row["DPE_NAME"].data<std::string>();
    std::string an = row["ALIAS"   ].data<std::string>();
    cnMap.insert( std::pair<std::string,std::string>( dp, an ) );
  }

  std::map<int,std::string>::const_iterator idIter = idMap.begin();
  std::map<int,std::string>::const_iterator idIend = idMap.end();
  std::string outChk( "/outputChannel" );
//  int lnum = 0;
  while ( idIter != idIend ) {
    const std::pair<int,std::string>& ientry = *idIter++;
//    lnum++;
    int dpId       = ientry.first;
    std::string dp = ientry.second;
    int ldp = dp.length();
    if ( ldp < 20 ) continue;
    std::string subOut( dp.substr( ldp - 17, 17 ) );
//    std::cout << dp << std::endl << subOut << std::endl;
    std::string subChk( subOut.substr( 0, 14 ) );
    if ( subChk != outChk ) continue;
//    std::cout << dp << std::endl << subOut << std::endl << subChk << std::endl;
    std::string chName( dp.substr( 0, ldp - 17 ) );
//    chName += ".actual.Trip";
    chName += ".actual.OvC";
    int chCode = subOut.c_str()[16] - '0';
//    std::cout << dp << std::endl << chName << " " << chCode << std::endl;
    std::map<std::string,std::string>::const_iterator jter =
                                                      cnMap.find( chName );
    if ( jter == cnMap.end() ) continue;
//    if ( jter == cnMap.end() ) {
//      std::cout << "***************" << std::endl;
//      break;
//    }
    const std::pair<std::string,std::string>& jentry = *jter;
//    std::cout << dp << std::endl << chName << " " << chCode << std::endl;
//    std::cout << jentry.second << std::endl;
    std::string an( jentry.second );
    int al = an.length();
    int iofw = 7 + an.find( "DT_HV_W", 0 );
    int iofc = 3 + an.find( "_MB", 0 );
    int iofs = 2 + an.find( "_S" , 0 );
    int iofq = 3 + an.find( "_SL", 0 );
    int iofl = 2 + an.find( "_L" , 0 );
    if ( ( iofw == al ) ||
         ( iofc == al ) ||
         ( iofs == al ) ||
         ( iofq == al ) ||
         ( iofl == al ) ) {
      std::cout << "***************" << std::endl;
      break;
    }
    int ioew = an.find( "_", iofw );
    int ioec = an.find( "_", iofc );
    int ioes = an.find( "_", iofs );
    int ioeq = an.find( "_", iofq );
    int ioel = an.find( "_", iofl );
//    std::cout << an.substr( iofw, ioew - iofw ) << " "
//              << an.substr( iofc, ioec - iofc ) << " "
//              << an.substr( iofs, ioes - iofs ) << " "
//              << an.substr( iofq, ioeq - iofq ) << " "
//              << an.substr( iofl, ioel - iofl ) << std::endl;
    std::string swhe( an.substr( iofw, ioew - iofw ) );
    const char* cwhe = swhe.c_str();
    int whe = cwhe[1] - '0';
    if ( *cwhe != 'P' ) whe = -whe;

    std::string scha( an.substr( iofc, ioec - iofc ) );
    const char* ccha = scha.c_str();
    int cha = *ccha - '0';

    std::string ssec( an.substr( iofs, ioes - iofs ) );
    const char* csec = ssec.c_str();
    int sec = ( ( *csec - '0' ) * 10 ) + ( csec[1] - '0' );
    if ( ( csec[2] == 'R' ) && ( sec == 10 ) ) sec = 14;
    if ( ( csec[2] == 'L' ) && ( sec ==  4 ) ) sec = 13;

    std::string squa( an.substr( iofq, ioeq - iofq ) );
    const char* cqua = squa.c_str();
    int qua = *cqua - '0';

    std::string slay( an.substr( iofl, ioel - iofl ) );
    const char* clay = slay.c_str();
    int lay = *clay - '0';

    std::cout << whe << " "
              << cha << " "
              << sec << " "
              << qua << " "
              << lay << " "
              << chCode << " "
              << dpId << std::endl;

/*
*/
    DTWireId wireId( whe, cha, sec, qua, lay, 10 + chCode );
    int chId = wireId.rawId();
    coral::AttributeList newChan;
    newChan.extend( "DETID", typeid(int) );
    newChan.extend(  "DPID", typeid(int) );
    newChan["DETID"].data<int>() = chId;
    newChan[ "DPID"].data<int>() = dpId;
    hvalTable.dataEditor().insertRow( newChan );
    aliasMap.insert( std::pair<int,int>( dpId, chId ) );
//    break;
  }
//  std::string dpname = idMap.begin()->second;
//  std::cout << dpname << std::endl;

  std::cout << "DTHVStatusHandler::dumpHVAliases - end" << std::endl;
  return;
}


void DTHVStatusHandler::updateHVStatus( cond::Time_t time ) {

  //////// test read from OMDS table ////////

  coral::TimeStamp startTime( ySince, mSince, dSince, 1, 0, 0, 0 );
  coral::ITable& fwccTable =
    omds_s_proxy->nominalSchema().tableHandle( "FWCAENCHANNEL" );
  std::auto_ptr<coral::IQuery>
    fwccQuery( fwccTable.newQuery() );
  fwccQuery->addToOutputList( "CHANGE_DATE" );
//  fwccQuery->addToOrderList( "CHANGE_DATE DESC" );
  coral::AttributeList timeBindVariableList;
  timeBindVariableList.extend( "since", typeid(coral::TimeStamp) );
  timeBindVariableList["since"].data<coral::TimeStamp>() =
          coral::TimeStamp( yUntil, mUntil, dUntil, 0, 0, 0, 0 );
//  fwccQuery->setCondition( "CHANGE_DATE>:since", timeBindVariableList );
  coral::ICursor& fwccCursor = fwccQuery->execute();
  if ( fwccCursor.next() )
       startTime = fwccCursor.currentRow()["CHANGE_DATE"].data<coral::TimeStamp>();

  ///////////////////////////////////////////

//  coral::TimeStamp::ValueType vts( 1224968952862390000LL );
//  coral::TimeStamp coralTime( yUntil, mUntil, dUntil, 2, 50, 49, 862390 );
//  coral::TimeStamp coralTime( yUntil, mUntil, dUntil, 21, 9, 12, 862390 );
  coral::TimeStamp coralTime( 2009, 2, 17, 16, 36, 11, 12345 );
//  coral::TimeStamp coralTime( yUntil, mUntil, dUntil, 21, 9, 13, 0 );
//  coral::TimeStamp coralTime( vts );

  std::cout << "CORAL until: " << coralTime.total_nanoseconds() << std::endl;
  std::cout << "CORAL until: " << 
    ((coralTime.total_nanoseconds()/1000000000)<<32) << std::endl;
  std::cout << "CORAL until: " << 
    ((coralTime.total_nanoseconds()/1000000000)<<32) +
    ((coralTime.total_nanoseconds()%1000000000)/1000)<< std::endl;
  std::cout << "CORAL until: " << 
    ((coralTime.total_nanoseconds()/1000000   )<<32) +
     (coralTime.total_nanoseconds()%1000000   )      << std::endl;
  unsigned long long iov_s = coralTime.total_nanoseconds()/1000000000;
  int iov_u = coralTime.total_nanoseconds()%1000000000;
//  iov_u /= 1000;
//  cond::Time_t iovtime = ((coralTime.total_nanoseconds()/1000000000)<<32) +
//                         ((coralTime.total_nanoseconds()%1000000000)/1000);
//  cond::Time_t iovtime = (iov_s<<32) + iov_u;
  cond::Time_t iovtime = ( iov_s << 32 ) + ( iov_u / 1000 );
  std::cout << "IOV until: " << iovtime << std::endl;

  edm::TimeValue_t daqtime=0LL;
  ::timeval stv;
  gettimeofday(&stv,0);
  daqtime=stv.tv_sec;
//  daqtime=(daqtime<<32)+stv.tv_usec;
  daqtime=(daqtime*1000000000)+stv.tv_usec;
  edm::Timestamp daqstamp(daqtime);
  std::cout << "edm until: " << daqstamp.value() << std::endl;

  ///////////////////////////////////////////

  //////// test insert into buffer table ////////

  std::cout << "test open snapshot table..." << std::endl;
  coral::ITable& bufferTable = 
    buff_s_proxy->nominalSchema().tableHandle( "HVSNAPSHOT" );
  std::cout << "test create snapshot data..." << std::endl;
  coral::AttributeList newMeas;
  newMeas.extend( "TIME",       typeid(coral::TimeStamp) );
  newMeas.extend( "WHEEL",      typeid(int) );
  newMeas.extend( "STATION",    typeid(int) );
  newMeas.extend( "SECTOR",     typeid(int) );
  newMeas.extend( "SUPERLAYER", typeid(int) );
  newMeas.extend( "LAYER",      typeid(int) );
  newMeas.extend( "PART",       typeid(int) );
  newMeas.extend( "STATUS",     typeid(int) );
  std::cout << "test write snapshot data..." << std::endl;
  newMeas["TIME"      ].data<coral::TimeStamp>() = startTime;
//  newMeas["TIME"      ].data<coral::TimeStamp>() = coralTime;
  newMeas["WHEEL"     ].data<int>() = 1;
  newMeas["STATION"   ].data<int>() = 2;
  newMeas["SECTOR"    ].data<int>() = 3;
  newMeas["SUPERLAYER"].data<int>() = 1;
  newMeas["LAYER"     ].data<int>() = 2;
  newMeas["PART"      ].data<int>() = 0;
  newMeas["STATUS"    ].data<int>() = 999;
  std::cout << "test fill snapshot table..." << std::endl;
  bufferTable.dataEditor().insertRow( newMeas );

  ///////////////////////////////////////////////
/*
*/
/*
// test buffer write
  std::map<int,float> hvMap;
  int iDum;
  for ( iDum = 1; iDum < 5; iDum++ ) {
    if ( hvMap.find( iDum ) != hvMap.end() ) continue;
    hvMap.insert( std::pair<int,float>( iDum, iDum * 1.1 ) );
  }

  std::vector<int> bufferList;
  coral::ITable& bufferTable = 
    buff_s_proxy->nominalSchema().tableHandle( "HVSNAPSHOT" );
  std::auto_ptr<coral::IQuery>
    snapshotQuery( bufferTable.newQuery() );
  snapshotQuery->addToOutputList( "Id" );
  coral::ICursor& snapshotCursor = snapshotQuery->execute();
  while( snapshotCursor.next() ) {
    const coral::AttributeList& row = snapshotCursor.currentRow();
    bufferList.push_back( row["Id"].data<int>() );
  }

  std::vector<int>::const_iterator id_iter = bufferList.begin();
  std::vector<int>::const_iterator id_iend = bufferList.end();
  while ( id_iter != id_iend ) {
    int currentId = *id_iend++;
    std::map<int,float>::iterator
    hv_iter = hvMap.find( currentId );
    if ( hv_iter == hvMap.end() ) continue;
    float value = hv_iter->second;
    coral::AttributeList newMeas;
    newMeas.extend( "id",    typeid(int  ) );
    newMeas.extend( "value", typeid(float) );
    newMeas["Id"     ].data<int>()   = currentId;
    newMeas["idValue"].data<float>() = value;
    bufferTable.dataEditor().updateRows( "Value=:value",
			                 "Id=:id", newMeas );
    hvMap.erase( hv_iter );
  }
  std::map<int,float>::const_iterator hv_iter = hvMap.begin();
  std::map<int,float>::const_iterator hv_iend = hvMap.end();
  while ( hv_iter != hv_iend ) {
    const std::pair<int,float>& hvEntry = *hv_iter++;
    coral::AttributeList newMeas;
    newMeas.extend( "id",    typeid(int  ) );
    newMeas.extend( "value", typeid(float) );
    newMeas["Id"     ].data<int>()   = hvEntry.first;
    newMeas["idValue"].data<float>() = hvEntry.second;
    bufferTable.dataEditor().insertRow( newMeas );
  }
*/
  return;

}
/*
*/


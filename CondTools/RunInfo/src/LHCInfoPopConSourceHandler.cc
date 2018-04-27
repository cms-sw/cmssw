#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondTools/RunInfo/interface/LHCInfoPopConSourceHandler.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

LHCInfoPopConSourceHandler::LHCInfoPopConSourceHandler( edm::ParameterSet const & pset ):
  m_debug( pset.getUntrackedParameter<bool>( "debug", false ) )
  ,m_startTime()
  ,m_endTime()
  ,m_samplingInterval( (unsigned int)pset.getUntrackedParameter<unsigned int>( "samplingInterval", 300 ) )
  ,m_name( pset.getUntrackedParameter<std::string>( "name", "LHCInfoPopConSourceHandler" ) )
  ,m_connectionString(pset.getUntrackedParameter<std::string>("connectionString",""))
  ,m_ecalConnectionString(pset.getUntrackedParameter<std::string>("ecalConnectionString",""))
  ,m_dipSchema(pset.getUntrackedParameter<std::string>("DIPSchema",""))
  ,m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath",""))
  ,m_payloadBuffer() {
  if( pset.exists("startTime") ){
    m_startTime = boost::posix_time::time_from_string( pset.getUntrackedParameter<std::string>("startTime" ) );
  }
  boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
  m_endTime = now;
  if( pset.exists("endTime") ){
    m_endTime = boost::posix_time::time_from_string( pset.getUntrackedParameter<std::string>("endTime" ) );
    if(m_endTime>now) m_endTime = now;
  } 
}
//L1: try with different m_dipSchema
//L2: try with different m_name
LHCInfoPopConSourceHandler::~LHCInfoPopConSourceHandler() {}

namespace LHCInfoImpl {
  cond::Time_t getNextIov( cond::Time_t prevIov, unsigned int samplingInterval ){
    cond::UnpackedTime ut = cond::time::unpack( prevIov );
    ut.first +=  samplingInterval;
    return cond::time::pack( ut );
  }
}

bool LHCInfoPopConSourceHandler::getFillData( cond::persistency::Session& session, 
					      const boost::posix_time::ptime& targetTime, 
					      bool next, 
					      LHCInfo& payload ){
  coral::ISchema& runTimeLoggerSchema = session.nominalSchema();
  //prepare the query for table 1:
  std::unique_ptr<coral::IQuery> fillDataQuery( runTimeLoggerSchema.newQuery() );
  //FROM clause
  fillDataQuery->addToTableList( std::string( "RUNTIME_SUMMARY" ) );
  //SELECT clause
  fillDataQuery->addToOutputList( std::string( "LHCFILL" ) );
  fillDataQuery->addToOutputList( std::string( "NBUNCHESBEAM1" ) );
  fillDataQuery->addToOutputList( std::string( "NBUNCHESBEAM2" ) );
  fillDataQuery->addToOutputList( std::string( "NCOLLIDINGBUNCHES" ) );
  fillDataQuery->addToOutputList( std::string( "NTARGETBUNCHES" ) );
  fillDataQuery->addToOutputList( std::string( "RUNTIME_TYPE_ID" ) );
  fillDataQuery->addToOutputList( std::string( "PARTY1" ) );
  fillDataQuery->addToOutputList( std::string( "PARTY2" ) );
  fillDataQuery->addToOutputList( std::string( "CROSSINGANGLE" ) );
  fillDataQuery->addToOutputList( std::string( "BETASTAR" ) );
  fillDataQuery->addToOutputList( std::string( "INTENSITYBEAM1" ) );
  fillDataQuery->addToOutputList( std::string( "INTENSITYBEAM2" ) );
  fillDataQuery->addToOutputList( std::string( "ENERGY" ) );
  fillDataQuery->addToOutputList( std::string( "CREATETIME" ) );
  fillDataQuery->addToOutputList( std::string( "BEGINTIME" ) );
  fillDataQuery->addToOutputList( std::string( "ENDTIME" ) );
  fillDataQuery->addToOutputList( std::string( "INJECTIONSCHEME" ) );
  //WHERE clause
  coral::AttributeList fillDataBindVariables;
  fillDataBindVariables.extend<coral::TimeStamp>(std::string("targetTime"));
  fillDataBindVariables[ std::string( "targetTime")].data<coral::TimeStamp>()= coral::TimeStamp( targetTime + boost::posix_time::seconds(1) ); 
  //by imposing BEGINTIME IS NOT NULL, we remove fills which never went into stable beams,
  //or the most recent one, just declared but not yet in stable beams
  std::string conditionStr( "BEGINTIME IS NOT NULL AND BEGINTIME <= :targetTime AND ENDTIME> :targetTime AND LHCFILL IS NOT NULL" );
  if( next ){
    conditionStr = "BEGINTIME IS NOT NULL AND BEGINTIME > :targetTime AND LHCFILL IS NOT NULL AND ENDTIME IS NOT NULL";
  }
  fillDataQuery->setCondition( conditionStr, fillDataBindVariables );
  //ORDER BY clause
  fillDataQuery->addToOrderList( std::string( "BEGINTIME" ) );
  //define query output  
  coral::AttributeList fillDataOutput;
  fillDataOutput.extend<unsigned short>( std::string( "LHCFILL" ) );
  fillDataOutput.extend<unsigned short>( std::string( "NBUNCHESBEAM1" ) );
  fillDataOutput.extend<unsigned short>( std::string( "NBUNCHESBEAM2" ) );
  fillDataOutput.extend<unsigned short>( std::string( "NCOLLIDINGBUNCHES" ) );
  fillDataOutput.extend<unsigned short>( std::string( "NTARGETBUNCHES" ) );
  fillDataOutput.extend<int>( std::string( "RUNTIME_TYPE_ID" ) );
  fillDataOutput.extend<int>( std::string( "PARTY1" ) );
  fillDataOutput.extend<int>( std::string( "PARTY2" ) );
  fillDataOutput.extend<float>( std::string( "CROSSINGANGLE" ) );
  fillDataOutput.extend<float>( std::string( "BETASTAR" ) );
  fillDataOutput.extend<float>( std::string( "INTENSITYBEAM1" ) );
  fillDataOutput.extend<float>( std::string( "INTENSITYBEAM2" ) );
  fillDataOutput.extend<float>( std::string( "ENERGY" ) );
  fillDataOutput.extend<coral::TimeStamp>( std::string( "CREATETIME" ) );
  fillDataOutput.extend<coral::TimeStamp>( std::string( "BEGINTIME" ) );
  fillDataOutput.extend<coral::TimeStamp>( std::string( "ENDTIME" ) );
  fillDataOutput.extend<std::string>( std::string( "INJECTIONSCHEME" ) );
  fillDataQuery->defineOutput( fillDataOutput );
  fillDataQuery->limitReturnedRows( 1 );
  //execute the query
  coral::ICursor& fillDataCursor = fillDataQuery->execute();
  //
  unsigned short currentFill = 0;
  unsigned short bunches1 = 0, bunches2 = 0, collidingBunches = 0, targetBunches = 0;
  LHCInfo::FillTypeId fillType = LHCInfo::UNKNOWN;
  LHCInfo::ParticleTypeId particleType1 = LHCInfo::NONE, particleType2 = LHCInfo::NONE;
  float crossingAngle = 0., betastar = 0., intensityBeam1 = 0., intensityBeam2 = 0., energy = 0.;
  coral::TimeStamp stableBeamStartTimeStamp, beamDumpTimeStamp;
  cond::Time_t creationTime = 0ULL, stableBeamStartTime = 0ULL, beamDumpTime = 0ULL;
  std::string injectionScheme( "None" );
  std::ostringstream ss;
  bool ret = false;
  if( fillDataCursor.next() ) {
    ret = true;
    if( m_debug ) {
      std::ostringstream qs;
      fillDataCursor.currentRow().toOutputStream( qs );
    }
    coral::Attribute const & fillAttribute = fillDataCursor.currentRow()[ std::string( "LHCFILL" ) ];
    if( !fillAttribute.isNull() ){
      currentFill = fillAttribute.data<unsigned short>();
    }
    coral::Attribute const & bunches1Attribute = fillDataCursor.currentRow()[ std::string( "NBUNCHESBEAM1" ) ];
    if( !bunches1Attribute.isNull() ) {
      bunches1 = bunches1Attribute.data<unsigned short>();
    }
    coral::Attribute const & bunches2Attribute = fillDataCursor.currentRow()[ std::string( "NBUNCHESBEAM2" ) ];
    if( !bunches2Attribute.isNull() ) {
      bunches2 = bunches2Attribute.data<unsigned short>();
    }
    coral::Attribute const & collidingBunchesAttribute = fillDataCursor.currentRow()[ std::string( "NCOLLIDINGBUNCHES" ) ];
    if( !collidingBunchesAttribute.isNull() ) {
      collidingBunches = collidingBunchesAttribute.data<unsigned short>();
    }
    coral::Attribute const & targetBunchesAttribute = fillDataCursor.currentRow()[ std::string( "NTARGETBUNCHES" ) ];
    if( !targetBunchesAttribute.isNull() ) {
      targetBunches = targetBunchesAttribute.data<unsigned short>();
    }
    //RUNTIME_TYPE_ID IS NOT NULL
    fillType = static_cast<LHCInfo::FillTypeId>( fillDataCursor.currentRow()[ std::string( "RUNTIME_TYPE_ID" ) ].data<int>() );
    coral::Attribute const & particleType1Attribute = fillDataCursor.currentRow()[ std::string( "PARTY1" ) ];
    if( !particleType1Attribute.isNull() ) {
      particleType1 = static_cast<LHCInfo::ParticleTypeId>( particleType1Attribute.data<int>() );
    }
    coral::Attribute const & particleType2Attribute = fillDataCursor.currentRow()[ std::string( "PARTY2" ) ];
    if( !particleType2Attribute.isNull() ) {
      particleType2 = static_cast<LHCInfo::ParticleTypeId>( particleType2Attribute.data<int>() );
    }
    coral::Attribute const & crossingAngleAttribute = fillDataCursor.currentRow()[ std::string( "CROSSINGANGLE" ) ];
    if( !crossingAngleAttribute.isNull() ) {
      crossingAngle = crossingAngleAttribute.data<float>();
    }
    coral::Attribute const & betastarAttribute = fillDataCursor.currentRow()[ std::string( "BETASTAR" ) ];
    if( !betastarAttribute.isNull() ) {
      betastar = betastarAttribute.data<float>();
    }
    coral::Attribute const & intensityBeam1Attribute = fillDataCursor.currentRow()[ std::string( "INTENSITYBEAM1" ) ];
    if( !intensityBeam1Attribute.isNull() ) {
      intensityBeam1 = intensityBeam1Attribute.data<float>();
    }
    coral::Attribute const & intensityBeam2Attribute = fillDataCursor.currentRow()[ std::string( "INTENSITYBEAM2" ) ];
    if( !intensityBeam2Attribute.isNull() ) {
      intensityBeam2 = intensityBeam2Attribute.data<float>();
    }
    coral::Attribute const & energyAttribute = fillDataCursor.currentRow()[ std::string( "ENERGY" ) ];
    if( !energyAttribute.isNull() ){
      energy = energyAttribute.data<float>();
    }
  }
  if( ret ){
    //CREATETIME IS NOT NULL
    creationTime = cond::time::from_boost( fillDataCursor.currentRow()[ std::string( "CREATETIME" ) ].data<coral::TimeStamp>().time() );
    //BEGINTIME is imposed to be NOT NULL in the WHERE clause
    stableBeamStartTimeStamp = fillDataCursor.currentRow()[ std::string( "BEGINTIME" ) ].data<coral::TimeStamp>();
    stableBeamStartTime = cond::time::from_boost( stableBeamStartTimeStamp.time() );
    coral::Attribute const & beamDumpTimeAttribute = fillDataCursor.currentRow()[ std::string( "ENDTIME" ) ];
    if( !beamDumpTimeAttribute.isNull() ) {
      beamDumpTimeStamp = beamDumpTimeAttribute.data<coral::TimeStamp>();
      beamDumpTime = cond::time::from_boost( beamDumpTimeStamp.time() );
    }
    coral::Attribute const & injectionSchemeAttribute = fillDataCursor.currentRow()[ std::string( "INJECTIONSCHEME" ) ];
    if( !injectionSchemeAttribute.isNull() ) {
      injectionScheme = injectionSchemeAttribute.data<std::string>();
    }
    //fix an inconsistency in RunTimeLogger: if the fill type is defined, the particle type should reflect it!
    if( fillType != LHCInfo::UNKNOWN && ( particleType1 == LHCInfo::NONE || particleType2 == LHCInfo::NONE ) ) {
      switch( fillType ) {
      case LHCInfo::PROTONS :
	particleType1 = LHCInfo::PROTON;
	particleType2 = LHCInfo::PROTON;
	break;
      case LHCInfo::IONS :
	particleType1 = LHCInfo::PB82;
	particleType2 = LHCInfo::PB82;
	break;
      case LHCInfo::UNKNOWN :
      case LHCInfo::COSMICS :
      case LHCInfo::GAP :
	break;
      }
    }
    payload.setFill( currentFill, true );
    payload.setBunchesInBeam1( bunches1 );
    payload.setBunchesInBeam2( bunches2 );
    payload.setCollidingBunches( collidingBunches );
    payload.setTargetBunches( targetBunches );
    payload.setFillType( fillType );
    payload.setParticleTypeForBeam1( particleType1 );
    payload.setParticleTypeForBeam2( particleType2 );
    payload.setCrossingAngle( crossingAngle );
    payload.setBetaStar( betastar );
    payload.setIntensityForBeam1( intensityBeam1 );
    payload.setIntensityForBeam2( intensityBeam2 );
    payload.setEnergy( energy );
    payload.setCreationTime( creationTime );
    payload.setBeginTime( stableBeamStartTime );
    payload.setEndTime( beamDumpTime );
    payload.setInjectionScheme( injectionScheme );
  }
  return ret;
}

bool LHCInfoPopConSourceHandler::getCurrentFillData( cond::persistency::Session& session,
						     const boost::posix_time::ptime& targetTime,
						     LHCInfo& payload ){
  return getFillData( session, targetTime, false, payload );
}

bool LHCInfoPopConSourceHandler::getNextFillData( cond::persistency::Session& session,
						  const boost::posix_time::ptime& targetTime,
						  LHCInfo& payload ){
  return getFillData( session, targetTime, true, payload );
}

bool LHCInfoPopConSourceHandler::getLumiData( cond::persistency::Session& session, 
					      const boost::posix_time::ptime& targetTime, 
					      LHCInfo& payload ){
  coral::ISchema& runTimeLoggerSchema = session.nominalSchema();
  //prepare the query for table 2:
  std::unique_ptr<coral::IQuery> fillDataQuery2( runTimeLoggerSchema.newQuery() );
  //FROM clause
  fillDataQuery2->addToTableList( std::string( "LUMI_SECTIONS" ) );
  //SELECT clause
  fillDataQuery2->addToOutputList( std::string( "DELIVLUMI" ) );
  fillDataQuery2->addToOutputList( std::string( "LIVELUMI" ) );
  fillDataQuery2->addToOutputList( std::string( "INSTLUMI" ) );
  fillDataQuery2->addToOutputList( std::string( "INSTLUMIERROR" ) );
  //WHERE clause
  coral::AttributeList fillDataBindVariables;
  fillDataBindVariables.extend<coral::TimeStamp>(std::string("targetTime"));
  fillDataBindVariables[ std::string( "targetTime")].data<coral::TimeStamp>()= coral::TimeStamp( targetTime + boost::posix_time::seconds(1) ); 
  std::string conditionStr = "DELIVLUMI IS NOT NULL AND STARTTIME < :targetTime AND STOPTIME> :targetTime";
  fillDataQuery2->setCondition( conditionStr, fillDataBindVariables );
  //ORDER BY clause
  fillDataQuery2->addToOrderList( std::string( "LHCFILL" ) );
  //fillDataQuery2->groupBy( std::string( "LHCFILL" ) );
  //define query output*/
  coral::AttributeList fillDataOutput2;
  fillDataOutput2.extend<float>( std::string( "DELIVEREDLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "RECORDEDLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "INSTLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "INSTLUMIERROR" ) );
  fillDataQuery2->defineOutput( fillDataOutput2 );
  //execute the query
  coral::ICursor& fillDataCursor2 = fillDataQuery2->execute();
  
  float delivLumi = 0., recLumi = 0., instLumi = 0, instLumiErr = 0.;
  bool ret = false;
  if( fillDataCursor2.next()){
    ret = true;
    coral::Attribute const & delivLumiAttribute = fillDataCursor2.currentRow()[ std::string( "DELIVEREDLUMI" ) ];
    if( !delivLumiAttribute.isNull() ){
      delivLumi = delivLumiAttribute.data<float>() / 1000.;
    }
    coral::Attribute const & recLumiAttribute = fillDataCursor2.currentRow()[ std::string( "RECORDEDLUMI" ) ];
    if( !recLumiAttribute.isNull() ){
      recLumi = recLumiAttribute.data<float>() / 1000.;
    }
    coral::Attribute const & instLumiAttribute = fillDataCursor2.currentRow()[ std::string( "INSTLUMI" ) ];
    if( !instLumiAttribute.isNull() ){
      instLumi = instLumiAttribute.data<float>() / 1000.;
    }
    coral::Attribute const & instLumiErrAttribute = fillDataCursor2.currentRow()[ std::string( "INSTLUMIERROR" ) ];
    if( !instLumiErrAttribute.isNull() ){
      instLumiErr = instLumiErrAttribute.data<float>() / 1000.;
    }
    if( delivLumi > 0. ){
      payload.setDelivLumi( delivLumi );
      payload.setRecLumi( recLumi );
      payload.setInstLumi( instLumi );
      payload.setInstLumiError( instLumiErr );
    }
  }
  return ret;  
}

 bool LHCInfoPopConSourceHandler::getDipData( cond::persistency::Session& session, 
					      const boost::posix_time::ptime& targetTime, 
					      LHCInfo& payload ){
     //run the third and fourth query against the schema hosting detailed DIP information
   coral::ISchema& beamCondSchema = session.coralSession().schema( m_dipSchema );
   //start the transaction against the DIP "deep" database backend schema
   //prepare the WHERE clause for both queries
   coral::AttributeList bunchConfBindVariables;
   bunchConfBindVariables.extend<coral::TimeStamp>(std::string("targetTime"));
   bunchConfBindVariables[ std::string( "targetTime")].data<coral::TimeStamp>()= coral::TimeStamp( targetTime + boost::posix_time::seconds(1) ); 
   std::string conditionStr = std::string( "DIPTIME <= :targetTime" );
   //define the output types for both queries
   coral::AttributeList bunchConfOutput;
   bunchConfOutput.extend<coral::TimeStamp>( std::string( "DIPTIME" ) );
   bunchConfOutput.extend<unsigned short>( std::string( "BUCKET" ) );
   //execute query for Beam 1
   std::unique_ptr<coral::IQuery> bunchConf1Query(beamCondSchema.newQuery());
   bunchConf1Query->addToTableList( std::string( "LHC_CIRCBUNCHCONFIG_BEAM1" ), std::string( "BEAMCONF\", TABLE( BEAMCONF.VALUE ) \"BUCKETS" ) );
   bunchConf1Query->addToOutputList( std::string( "BEAMCONF.DIPTIME" ), std::string( "DIPTIME" ) );
   bunchConf1Query->addToOutputList( std::string( "BUCKETS.COLUMN_VALUE" ), std::string( "BUCKET" ) );
   bunchConf1Query->setCondition( conditionStr, bunchConfBindVariables );
   bunchConf1Query->addToOrderList( std::string( "DIPTIME DESC" ) );
   bunchConf1Query->limitReturnedRows( LHCInfo::availableBunchSlots ); //maximum number of filled bunches
   bunchConf1Query->defineOutput( bunchConfOutput );
   coral::ICursor& bunchConf1Cursor = bunchConf1Query->execute();

   std::bitset<LHCInfo::bunchSlots+1> bunchConfiguration1( 0ULL );   
   bool ret = false;
   while( bunchConf1Cursor.next() ) {
     ret = true;
     if( m_debug ) {
       std::ostringstream b1s;
       bunchConf1Cursor.currentRow().toOutputStream( b1s );
     }
     if( bunchConf1Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() != 0 ) {
       unsigned short slot = ( bunchConf1Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() - 1 ) / 10 + 1;
       bunchConfiguration1[ slot ] = true;
     }
   }
   
   if(ret){
     payload.setBunchBitsetForBeam1( bunchConfiguration1 );
   }
   
   //execute query for Beam 2
   std::unique_ptr<coral::IQuery> bunchConf2Query(beamCondSchema.newQuery());
   bunchConf2Query->addToTableList( std::string( "LHC_CIRCBUNCHCONFIG_BEAM2" ), std::string( "BEAMCONF\", TABLE( BEAMCONF.VALUE ) \"BUCKETS" ) );
   bunchConf2Query->addToOutputList( std::string( "BEAMCONF.DIPTIME" ), std::string( "DIPTIME" ) );
   bunchConf2Query->addToOutputList( std::string( "BUCKETS.COLUMN_VALUE" ), std::string( "BUCKET" ) );
   bunchConf2Query->setCondition( conditionStr, bunchConfBindVariables );
   bunchConf2Query->addToOrderList( std::string( "DIPTIME DESC" ) );
   bunchConf2Query->limitReturnedRows( LHCInfo::availableBunchSlots ); //maximum number of filled bunches
   bunchConf2Query->defineOutput( bunchConfOutput );
   coral::ICursor& bunchConf2Cursor = bunchConf2Query->execute();

   std::bitset<LHCInfo::bunchSlots+1> bunchConfiguration2( 0ULL );
   ret = false;
   while( bunchConf2Cursor.next() ) {
     ret = true;
     if( m_debug ) {
       std::ostringstream b2s;
       bunchConf2Cursor.currentRow().toOutputStream( b2s );
     }
     if( bunchConf2Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() != 0 ) {
       unsigned short slot = ( bunchConf2Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() - 1 ) / 10 + 1;
       bunchConfiguration2[ slot ] = true;
     }
   }
   if(ret){
     payload.setBunchBitsetForBeam2( bunchConfiguration2 );
   }
   
   //execute query for lumiPerBX
   std::unique_ptr<coral::IQuery> lumiDataQuery(beamCondSchema.newQuery());
   lumiDataQuery->addToTableList( std::string( "CMS_LHC_LUMIPERBUNCH" ), std::string( "LUMIPERBUNCH\", TABLE( LUMIPERBUNCH.LUMI_BUNCHINST ) \"VALUE" ) );
   lumiDataQuery->addToOutputList( std::string( "LUMIPERBUNCH.DIPTIME" ), std::string( "DIPTIME" ) );
   lumiDataQuery->addToOutputList( std::string( "VALUE.COLUMN_VALUE" ), std::string( "LUMI/BUNCH" ) );
   coral::AttributeList lumiDataBindVariables;
   lumiDataBindVariables.extend<coral::TimeStamp>( std::string( "targetTime" ) );
   lumiDataBindVariables[ std::string( "targetTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp( targetTime + boost::posix_time::seconds(1) );
   lumiDataBindVariables.extend<coral::TimeStamp>( std::string( "beamDumpTimeStamp" ) );
   lumiDataBindVariables[ std::string( "beamDumpTimeStamp" ) ].data<coral::TimeStamp>() = coral::TimeStamp(cond::time::to_boost(payload.endTime()));
   conditionStr = std::string( "DIPTIME BETWEEN :targetTime AND :beamDumpTimeStamp" );
   lumiDataQuery->setCondition( conditionStr, lumiDataBindVariables );
   lumiDataQuery->addToOrderList( std::string( "DIPTIME DESC" ) );
   lumiDataQuery->limitReturnedRows(3564); //Maximum number of bunches.
   //define query output
   coral::AttributeList lumiDataOutput;
   lumiDataOutput.extend<coral::TimeStamp>( std::string( "TIME" ) );
   lumiDataOutput.extend<float>( std::string( "VALUE" ) );
   lumiDataQuery->defineOutput( lumiDataOutput );
   //execute the query
   coral::ICursor& lumiDataCursor = lumiDataQuery->execute();

   std::vector<float> lumiPerBX;
   ret = false;
   while( lumiDataCursor.next() ) {
     ret = true;
     if( m_debug ) {
       std::ostringstream lpBX;
       lumiDataCursor.currentRow().toOutputStream( lpBX );
     }
     if( lumiDataCursor.currentRow()[ std::string( "VALUE" ) ].data<float>() != 0.00 ) {
       lumiPerBX.push_back(lumiDataCursor.currentRow()[ std::string( "VALUE" ) ].data<float>());
     }
   } 
   if( ret){
     payload.setLumiPerBX( lumiPerBX );
   }
   return ret;
 }

 bool LHCInfoPopConSourceHandler::getCTTPSData( cond::persistency::Session& session, 
						const boost::posix_time::ptime& targetTime, 
						LHCInfo& payload ){
   //run the fifth query against the CTPPS schema
   //Initializing the CMS_CTP_CTPPS_COND schema.
   coral::ISchema& CTPPS = session.coralSession().schema("CMS_CTP_CTPPS_COND");
   //execute query for CTPPS Data
   std::unique_ptr<coral::IQuery> CTPPSDataQuery( CTPPS.newQuery() );
   //FROM clause
   CTPPSDataQuery->addToTableList( std::string( "CTPPS_LHC_MACHINE_PARAMS" ) );
   //SELECT clause
   CTPPSDataQuery->addToOutputList( std::string( "LHC_STATE" ) );
   CTPPSDataQuery->addToOutputList( std::string( "LHC_COMMENT" ) );
   CTPPSDataQuery->addToOutputList( std::string( "CTPPS_STATUS" ) );
   CTPPSDataQuery->addToOutputList( std::string( "LUMI_SECTION" ) );
   //WHERE CLAUSE
   coral::AttributeList CTPPSDataBindVariables;
   CTPPSDataBindVariables.extend<coral::TimeStamp>( std::string( "targetTime" ) );
   CTPPSDataBindVariables[ std::string( "targetTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp( targetTime + boost::posix_time::seconds(1));
   std::string conditionStr = std::string( "DIP_UPDATE_TIME<= :targetTime" );
   CTPPSDataQuery->setCondition( conditionStr, CTPPSDataBindVariables );
    //ORDER BY clause
   CTPPSDataQuery->addToOrderList( std::string( "DIP_UPDATE_TIME DESC" ) ); //Only the latest value is fetched.
   //define query output
   coral::AttributeList CTPPSDataOutput;
   CTPPSDataOutput.extend<std::string>( std::string( "LHC_STATE" ) );
   CTPPSDataOutput.extend<std::string>( std::string( "LHC_COMMENT" ) );
   CTPPSDataOutput.extend<std::string>( std::string( "CTPPS_STATUS" ) );
   CTPPSDataOutput.extend<int>( std::string( "LUMI_SECTION" ) );
   CTPPSDataQuery->limitReturnedRows( 1 ); //Only one entry per payload.
   CTPPSDataQuery->defineOutput( CTPPSDataOutput );
   //execute the query
   coral::ICursor& CTPPSDataCursor = CTPPSDataQuery->execute();
   std::string lhcState = "", lhcComment = "", ctppsStatus = "";
   unsigned int lumiSection = 0;
   
   bool ret = false;
   if( CTPPSDataCursor.next() ) {
     ret = true;
     if( m_debug ) {
       std::ostringstream CTPPS;
       CTPPSDataCursor.currentRow().toOutputStream( CTPPS );
     }
     coral::Attribute const & lhcStateAttribute = CTPPSDataCursor.currentRow()[ std::string( "LHC_STATE" ) ];
     if( !lhcStateAttribute.isNull() ) {
       lhcState = lhcStateAttribute.data<std::string>();
     }
      
     coral::Attribute const & lhcCommentAttribute = CTPPSDataCursor.currentRow()[ std::string( "LHC_COMMENT" ) ];
     if( !lhcCommentAttribute.isNull() ) {
       lhcComment = lhcCommentAttribute.data<std::string>();
     }
      
     coral::Attribute const & ctppsStatusAttribute = CTPPSDataCursor.currentRow()[ std::string( "CTPPS_STATUS" ) ];
     if( !ctppsStatusAttribute.isNull() ) {
       ctppsStatus = ctppsStatusAttribute.data<std::string>();
     }
      
     coral::Attribute const & lumiSectionAttribute = CTPPSDataCursor.currentRow()[ std::string( "LUMI_SECTION" ) ];
     if( !lumiSectionAttribute.isNull() ) {
       lumiSection = lumiSectionAttribute.data<int>();
     }
     payload.setLhcState( lhcState );
     payload.setLhcComment( lhcComment );
     payload.setCtppsStatus( ctppsStatus );
     payload.setLumiSection( lumiSection );
   }
   return ret;
   
 }
bool LHCInfoPopConSourceHandler::getEcalData(  cond::persistency::Session& session, 
					       const boost::posix_time::ptime& targetTime, 
					       LHCInfo& payload ){
  //run the sixth query against the CMS_DCS_ENV_PVSS_COND schema
  //Initializing the CMS_DCS_ENV_PVSS_COND schema.
  coral::ISchema& ECAL = session.nominalSchema();
  //start the transaction against the fill logging schema
  //execute query for ECAL Data
  std::unique_ptr<coral::IQuery> ECALDataQuery( ECAL.newQuery() );
  //FROM clause
  ECALDataQuery->addToTableList( std::string( "BEAM_PHASE" ) );
  //SELECT clause 
  ECALDataQuery->addToOutputList( std::string( "DIP_value" ) );
  ECALDataQuery->addToOutputList( std::string( "element_nr" ) );
  //WHERE CLAUSE
  coral::AttributeList ECALDataBindVariables;
  std::string conditionStr = std::string( "DIP_value LIKE '%beamPhaseMean%' OR DIP_value LIKE '%cavPhaseMean%'" );
  
  ECALDataQuery->setCondition( conditionStr, ECALDataBindVariables );
  //ORDER BY clause
  ECALDataQuery->addToOrderList( std::string( "CHANGE_DATE" ) );
  ECALDataQuery->addToOrderList( std::string( "DIP_value" ) );
  ECALDataQuery->addToOrderList( std::string( "element_nr" ) );
  //define query output
  coral::AttributeList ECALDataOutput;
  ECALDataOutput.extend<std::string>( std::string( "DIP_value" ) );
  ECALDataOutput.extend<float>( std::string( "element_nr" ) );
  ECALDataQuery->limitReturnedRows( 14256 ); //3564 entries per vector.
  ECALDataQuery->defineOutput( ECALDataOutput );
  //execute the query
  coral::ICursor& ECALDataCursor = ECALDataQuery->execute();
  std::vector<float> beam1VC, beam2VC, beam1RF, beam2RF;
  std::string dipVal = "";
  std::map<std::string, int> vecMap;
  vecMap[std::string("Beam1/beamPhaseMean")] = 1;
  vecMap[std::string("Beam2/beamPhaseMean")] = 2;
  vecMap[std::string("Beam1/cavPhaseMean")] = 3;
  vecMap[std::string("Beam2/cavPhaseMean")] = 4;
  
  bool ret = false;
  while( ECALDataCursor.next() ) {
    ret = true;
    if( m_debug ) {
      std::ostringstream ECAL;
      ECALDataCursor.currentRow().toOutputStream( ECAL );
    }
    coral::Attribute const & dipValAttribute = ECALDataCursor.currentRow()[ std::string( "DIP_value" ) ];
    if( !dipValAttribute.isNull() ) {
      dipVal = dipValAttribute.data<std::string>();
    }
    
    coral::Attribute const & elementNrAttribute = ECALDataCursor.currentRow()[ std::string( "element_nr" ) ];
    if( !elementNrAttribute.isNull() ){
      switch( vecMap[dipVal] )
	{
	case 1:
	  beam1VC.push_back(elementNrAttribute.data<float>());
	  break;
	case 2:
	  beam2VC.push_back(elementNrAttribute.data<float>());
	  break;
	case 3:
	  beam1RF.push_back(elementNrAttribute.data<float>());
	  break;
	case 4:
	  beam2RF.push_back(elementNrAttribute.data<float>());
	  break;
	default:
	  break;
	}
    }
  }
  if( ret){
    payload.setBeam1VC(beam1VC);
    payload.setBeam2VC(beam2VC);
    payload.setBeam1RF(beam1RF);
    payload.setBeam2RF(beam2RF);
  }
  return ret;
}

void LHCInfoPopConSourceHandler::addEmptyPayload( cond::Time_t iov ){
  bool add = false;
  if( m_to_transfer.empty() ){
    if( !m_lastPayloadEmpty ) add = true;
  } else {
    LHCInfo* lastAdded = m_to_transfer.back().first;
    if( lastAdded->fillNumber() != 0 ) {
      add = true;
    }
  }
  if( add ) {
    LHCInfo* newPayload = new LHCInfo();
    m_payloadBuffer.emplace_back(newPayload);
    m_to_transfer.push_back( std::make_pair( newPayload, iov ) );
  }
}

void LHCInfoPopConSourceHandler::addPayload( LHCInfo& newPayload, cond::Time_t iov ){
  bool add = false;
  if( m_to_transfer.empty() ){
    add = true;
  } else {
    LHCInfo* lastAdded = m_to_transfer.back().first;
    if( !lastAdded->equals( newPayload ) ) {
      add = true;
    }
  }
  if( add ) {
    m_to_transfer.push_back( std::make_pair( &newPayload, iov ) );
  }
}


void LHCInfoPopConSourceHandler::getNewObjects() {
  //reference to the last payload in the tag
  Ref previousFill;
  
  //if a new tag is created, transfer fake fill from 1 to the first fill for the first time
  if ( tagInfo().name.empty() ) {
    edm::LogInfo( m_name ) << "New tag "<< tagInfo().name << "; from " << m_name << "::getNewObjects";
  } else {
    //check what is already inside the database
    edm::LogInfo( m_name ) << "got info for tag " << tagInfo().name 
			   << ", IOVSequence token " << tagInfo().token
			   << ": size " << tagInfo().size 
			   << ", last object valid since " << tagInfo().lastInterval.first 
			   << " ( "<< boost::posix_time::to_iso_extended_string( cond::time::to_boost( tagInfo().lastInterval.first ) )
			   << " ); from " << m_name << "::getNewObjects";
  }

  cond::Time_t lastIov = tagInfo().lastInterval.first; 
  if( lastIov == 0 ){
    // for a new or empty tag, an empty payload should be added on top with since=1
    addEmptyPayload( 1 );
  } else {
    edm::LogInfo( m_name ) << "The last Iov in tag " << tagInfo().name 
			   << " valid since " << lastIov
			   << "from " << m_name << "::getNewObjects";
  }

  cond::Time_t targetIov = LHCInfoImpl::getNextIov( lastIov, m_samplingInterval );
  if( !m_startTime.is_not_a_date_time() ){
    cond::Time_t tgtIov = cond::time::from_boost(m_startTime);
    if( targetIov < tgtIov ) targetIov = tgtIov;
  }

  cond::Time_t endIov = cond::time::MAX_VAL;
  if( !m_endTime.is_not_a_date_time() ){
    endIov = cond::time::from_boost(m_endTime);
  }
  edm::LogInfo(m_name) <<"Starting sampling at "<<boost::posix_time::to_simple_string(cond::time::to_boost(targetIov));
  std::unique_ptr<LHCInfo> currentFillPayload;
  
  //retrieve the data from the relational database source
  cond::persistency::ConnectionPool connection;
  //configure the connection
  if( m_debug ) {
    connection.setMessageVerbosity( coral::Debug );
  } else {
    connection.setMessageVerbosity( coral::Error );
  }
  connection.setAuthenticationPath( m_authpath );
  connection.configure();
  //create the sessions
  cond::persistency::Session session = connection.createSession( m_connectionString, false );
  cond::persistency::Session session2 = connection.createSession( m_ecalConnectionString, false );
  //start the transaction against the fill logging schema
  if( !tagInfo().lastPayloadToken.empty() ){
    cond::persistency::Session session3 = dbSession();
    session3.transaction().start(true);
    std::shared_ptr<LHCInfo> lastPayload = session3.fetchPayload<LHCInfo>( tagInfo().lastPayloadToken );
    session3.transaction().commit();
    if( lastPayload->fillNumber() != 0 ){
      currentFillPayload.reset(lastPayload->cloneFill());
    } else {
      m_lastPayloadEmpty = true;
    }
  }
  if( currentFillPayload.get() == nullptr ){
    currentFillPayload.reset( new LHCInfo() );
    session.transaction().start(true);
    bool foundFill = getCurrentFillData( session, cond::time::to_boost(targetIov ), *currentFillPayload );
    session.transaction().commit();
    if( foundFill ){
      edm::LogInfo( m_name ) <<"Found a fill at current time.";
      cond::Time_t firstIov = currentFillPayload->beginTime();
      firstIov = std::max( firstIov,lastIov );
      targetIov = std::max( targetIov,firstIov);
    } else { 
      edm::LogInfo( m_name ) <<"No fill found at current time."<<std::endl;
      currentFillPayload.reset();
      addEmptyPayload( targetIov );
      targetIov = LHCInfoImpl::getNextIov( targetIov, m_samplingInterval );
    }
  }

  while( true ){
    if( targetIov >= endIov ){
      edm::LogInfo( m_name ) <<"Sampling ended at the pre-setted time "<<boost::posix_time::to_simple_string(cond::time::to_boost( endIov ));
      break;
    }
    boost::posix_time::ptime targetTime = cond::time::to_boost( targetIov );
    if( !currentFillPayload.get() ){
      currentFillPayload.reset( new LHCInfo() ); 
      session.transaction().start(true);
      edm::LogInfo( m_name ) <<"Searching new fill after "<<boost::posix_time::to_simple_string(targetTime);
      bool foundFill = getNextFillData( session, targetTime, *currentFillPayload );
      session.transaction().commit();
      if ( !foundFill ){
	currentFillPayload.reset();
	edm::LogInfo( m_name )<<"No fill found...";
	addEmptyPayload( targetIov );
	break;
      }
      cond::Time_t newTargetIov = currentFillPayload->beginTime();
      edm::LogInfo( m_name ) <<"Found new fill at "<<boost::posix_time::to_simple_string(cond::time::to_boost(newTargetIov));
      if( newTargetIov > targetIov ){
	addEmptyPayload( targetIov );
	targetIov = newTargetIov;
      }
    } 
    bool more = true;
    while( more ){
      targetTime = cond::time::to_boost( targetIov );
      edm::LogInfo( m_name )<<"Getting sample at:"<<boost::posix_time::to_simple_string(targetTime);
      LHCInfo* payload =  currentFillPayload->cloneFill();
      m_payloadBuffer.emplace_back( payload );
      session.transaction().start(true);
      getLumiData( session, targetTime, *payload );
      getDipData( session, targetTime, *payload );
      getCTTPSData( session, targetTime, *payload );
      session.transaction().commit();
      session2.transaction().start(true);
      getEcalData( session2, targetTime, *payload );
      session2.transaction().commit();
      addPayload( *payload, targetIov );
      targetIov = LHCInfoImpl::getNextIov( targetIov, m_samplingInterval );
      cond::Time_t endSampling = currentFillPayload->endTime();
      if( endSampling == 0 ) endSampling = cond::time::from_boost( boost::posix_time::second_clock::local_time() ); 
      if( targetIov > endSampling ){
	edm::LogInfo( m_name )<<"End of sampling for current fill: endTime is "<<
	  boost::posix_time::to_simple_string(cond::time::to_boost(endSampling));
        targetIov = endSampling;
        currentFillPayload = nullptr;
	more = false;
      }
      if( targetIov >= endIov ){
	more = false;
      }
    }
  }
}

std::string LHCInfoPopConSourceHandler::id() const { 
  return m_name;
}

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
#include <cmath>

LHCInfoPopConSourceHandler::LHCInfoPopConSourceHandler( edm::ParameterSet const & pset ):
  m_debug( pset.getUntrackedParameter<bool>( "debug", false ) )
  ,m_startTime()
  ,m_endTime()
  ,m_samplingInterval( (unsigned int)pset.getUntrackedParameter<unsigned int>( "samplingInterval", 300 ) )
  ,m_endFill( pset.getUntrackedParameter<bool>( "endFill", true ) )
  ,m_name( pset.getUntrackedParameter<std::string>( "name", "LHCInfoPopConSourceHandler" ) )
  ,m_connectionString(pset.getUntrackedParameter<std::string>("connectionString",""))
  ,m_ecalConnectionString(pset.getUntrackedParameter<std::string>("ecalConnectionString",""))
  ,m_dipSchema(pset.getUntrackedParameter<std::string>("DIPSchema",""))
  ,m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath",""))
  ,m_fillPayload()
  ,m_prevPayload()
  ,m_tmpBuffer()
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

  struct IOVComp {    
    bool operator()( const cond::Time_t& x, const std::pair<cond::Time_t,std::shared_ptr<LHCInfo> >& y ){ return ( x < y.first ); }
  };
    
    // function to search in the vector the target time
  std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >::const_iterator search( const cond::Time_t& val, 
											  const std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> >>& container ){
    if( container.empty() ) return container.end();
    auto p = std::upper_bound( container.begin(), container.end(), val, IOVComp() );
    return (p!= container.begin()) ? p-1 : container.end();
  }

  bool makeFillDataQuery( cond::persistency::Session& session, 
			  const std::string& conditionString, 
			  const coral::AttributeList& fillDataBindVariables,
                          std::unique_ptr<LHCInfo>& targetPayload,
			  bool debug ){
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
    fillDataQuery->addToOutputList( std::string( "INTENSITYBEAM1" ) );
    fillDataQuery->addToOutputList( std::string( "INTENSITYBEAM2" ) );
    fillDataQuery->addToOutputList( std::string( "ENERGY" ) );
    fillDataQuery->addToOutputList( std::string( "CREATETIME" ) );
    fillDataQuery->addToOutputList( std::string( "BEGINTIME" ) );
    fillDataQuery->addToOutputList( std::string( "ENDTIME" ) );
    fillDataQuery->addToOutputList( std::string( "INJECTIONSCHEME" ) );
    //WHERE clause
    fillDataQuery->setCondition( conditionString, fillDataBindVariables );
    //ORDER BY clause
    std::string orderStr("BEGINTIME");
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
    float intensityBeam1 = 0., intensityBeam2 = 0., energy = 0.;
    coral::TimeStamp stableBeamStartTimeStamp, beamDumpTimeStamp;
    cond::Time_t creationTime = 0ULL, stableBeamStartTime = 0ULL, beamDumpTime = 0ULL;
    std::string injectionScheme( "None" );
    std::ostringstream ss;
    bool ret = false;
    if( fillDataCursor.next() ) {
      ret = true;
      if( debug ) {
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
      targetPayload.reset( new LHCInfo() );
      targetPayload->setFillNumber( currentFill );
      targetPayload->setBunchesInBeam1( bunches1 );
      targetPayload->setBunchesInBeam2( bunches2 );
      targetPayload->setCollidingBunches( collidingBunches );
      targetPayload->setTargetBunches( targetBunches );
      targetPayload->setFillType( fillType );
      targetPayload->setParticleTypeForBeam1( particleType1 );
      targetPayload->setParticleTypeForBeam2( particleType2 );
      targetPayload->setIntensityForBeam1( intensityBeam1 );
      targetPayload->setIntensityForBeam2( intensityBeam2 );
      targetPayload->setEnergy( energy );
      targetPayload->setCreationTime( creationTime );
      targetPayload->setBeginTime( stableBeamStartTime );
      targetPayload->setEndTime( beamDumpTime );
      targetPayload->setInjectionScheme( injectionScheme );
    }
    return ret;
  }
											  
}

bool LHCInfoPopConSourceHandler::getNextFillData( cond::persistency::Session& session, 
						  const boost::posix_time::ptime& targetTime,
						  bool ended ){
  // Prepare the WHERE clause
  coral::AttributeList fillDataBindVariables;
  fillDataBindVariables.extend<coral::TimeStamp>(std::string("targetTime"));
  fillDataBindVariables[ std::string( "targetTime")].data<coral::TimeStamp>()= coral::TimeStamp( targetTime + boost::posix_time::seconds(1) ); 
  //by imposing BEGINTIME IS NOT NULL, we remove fills which never went into stable beams,
  //by additionally imposing ENDTIME IS NOT NULL, we select only finished fills 
  std::string conditionStr = "BEGINTIME IS NOT NULL AND CREATETIME > :targetTime AND LHCFILL IS NOT NULL";
  if( ended )  conditionStr += " AND ENDTIME IS NOT NULL";
  return LHCInfoImpl::makeFillDataQuery( session, conditionStr, fillDataBindVariables, m_fillPayload,  m_debug );  
}

bool LHCInfoPopConSourceHandler::getFillData( cond::persistency::Session& session, 
					      unsigned short fillId ){
  // Prepare the WHERE clause
  coral::AttributeList fillDataBindVariables;
  fillDataBindVariables.extend<unsigned short>(std::string("fillId"));
  fillDataBindVariables[ std::string( "fillId")].data<unsigned short>()= fillId; 
  std::string conditionStr = "LHCFILL=:fillId";
  return LHCInfoImpl::makeFillDataQuery( session, conditionStr, fillDataBindVariables, m_fillPayload,  m_debug );    
}

size_t LHCInfoPopConSourceHandler::getLumiData( cond::persistency::Session& session, 
						const boost::posix_time::ptime& beginFillTime, 
						const boost::posix_time::ptime& endFillTime ){
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
  fillDataQuery2->addToOutputList( std::string( "STARTTIME" ) );
  fillDataQuery2->addToOutputList( std::string( "LHCFILL" ) );
  //WHERE clause
  coral::AttributeList fillDataBindVariables;
  fillDataBindVariables.extend<coral::TimeStamp>(std::string("start"));
  fillDataBindVariables.extend<coral::TimeStamp>(std::string("stop"));
  fillDataBindVariables[ std::string( "start")].data<coral::TimeStamp>()= coral::TimeStamp( beginFillTime ); 
  fillDataBindVariables[ std::string( "stop")].data<coral::TimeStamp>()= coral::TimeStamp( endFillTime ); 
  std::string conditionStr = "DELIVLUMI IS NOT NULL AND STARTTIME >= :start AND STARTTIME< :stop";
  fillDataQuery2->setCondition( conditionStr, fillDataBindVariables );
  //ORDER BY clause
  fillDataQuery2->addToOrderList( std::string( "STARTTIME" ) );
  //define query output*/
  coral::AttributeList fillDataOutput2;
  fillDataOutput2.extend<float>( std::string( "DELIVEREDLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "RECORDEDLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "INSTLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "INSTLUMIERROR" ) );
  fillDataOutput2.extend<coral::TimeStamp>( std::string( "STARTTIME" ) );
  fillDataOutput2.extend<int>( std::string( "LHCFILL" ) );
  fillDataQuery2->defineOutput( fillDataOutput2 );
  //execute the query
  coral::ICursor& fillDataCursor2 = fillDataQuery2->execute();
  
  size_t nlumi = 0;
  while( fillDataCursor2.next()){
    nlumi++;
    float delivLumi = 0., recLumi = 0., instLumi = 0, instLumiErr = 0.;
    cond::Time_t since = 0;
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
    coral::Attribute const & startLumiSectionAttribute = fillDataCursor2.currentRow()[ std::string( "STARTTIME" ) ];
    if( !startLumiSectionAttribute.isNull() ) {
      since = cond::time::from_boost( startLumiSectionAttribute.data<coral::TimeStamp>().time() );
    }
    LHCInfo* thisLumiSectionInfo = m_fillPayload->cloneFill();
    m_tmpBuffer.emplace_back(std::make_pair(since,thisLumiSectionInfo));
    LHCInfo& payload = *thisLumiSectionInfo;
    payload.setDelivLumi( delivLumi );
    payload.setRecLumi( recLumi );
    payload.setInstLumi( instLumi );
    payload.setInstLumiError( instLumiErr );
  }
  return nlumi;  
}
											 
namespace LHCInfoImpl {
  struct LumiSectionFilter {

    LumiSectionFilter( const std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >& samples ):
      currLow( samples.begin() ),
      currUp( samples.begin() ),
      end( samples.end() ){
      currUp++;
    }

    void reset( const std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >& samples ){
      currLow = samples.begin();
      currUp = samples.begin();
      currUp++;
      end = samples.end();
      currentDipTime = 0;
    }
    
    bool process( cond::Time_t dipTime ){
      if( currLow == end ) return false;
      bool search = false;
      if( currentDipTime == 0 ){
	search = true;
      } else {
        if( dipTime == currentDipTime ) return true;
	else {
	  cond::Time_t upper = cond::time::MAX_VAL;
          if(currUp != end ) upper = currUp->first;
	  if( dipTime < upper ) return false;
          else {
            search = true;
	  }
	}
      }
      if( search ){
	while(currUp != end and currUp->first < dipTime ){
	  currLow++;
	  currUp++;
	}
        currentDipTime = dipTime;
	return currLow != end;
      }
      return false;
    }

    cond::Time_t currentSince() {  return currLow->first; }
    LHCInfo& currentPayload() { return *currLow->second; }
    
    std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >::const_iterator current(){
      return currLow;
    }
    std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >::const_iterator currLow;
    std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >::const_iterator currUp;
    std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >::const_iterator end;
    cond::Time_t currentDipTime = 0;
  };
}

 bool LHCInfoPopConSourceHandler::getDipData( cond::persistency::Session& session, 
					      const boost::posix_time::ptime& beginFillTime,
					      const boost::posix_time::ptime& endFillTime ){
     //run the third and fourth query against the schema hosting detailed DIP information
   coral::ISchema& beamCondSchema = session.coralSession().schema( m_dipSchema );
   //start the transaction against the DIP "deep" database backend schema
   //prepare the WHERE clause for both queries
   coral::AttributeList bunchConfBindVariables;
   bunchConfBindVariables.extend<coral::TimeStamp>(std::string("beginFillTime"));
   bunchConfBindVariables.extend<coral::TimeStamp>(std::string("endFillTime"));
   bunchConfBindVariables[ std::string( "beginFillTime")].data<coral::TimeStamp>()= coral::TimeStamp( beginFillTime ); 
   bunchConfBindVariables[ std::string( "endFillTime")].data<coral::TimeStamp>()= coral::TimeStamp( endFillTime ); 
   std::string conditionStr = std::string( "DIPTIME >= :beginFillTime and DIPTIME< :endFillTime" );
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
   bunchConf1Query->addToOrderList( std::string( "DIPTIME" ) );
   bunchConf1Query->limitReturnedRows( LHCInfo::availableBunchSlots ); //maximum number of filled bunches
   bunchConf1Query->defineOutput( bunchConfOutput );

   coral::ICursor& bunchConf1Cursor = bunchConf1Query->execute();
   std::bitset<LHCInfo::bunchSlots+1> bunchConfiguration1( 0ULL ); 
   bool ret = false;
   cond::Time_t lumiSectionTime = 0;
   while( bunchConf1Cursor.next() ) {
     if( m_debug ) {
       std::ostringstream b1s;
       bunchConf1Cursor.currentRow().toOutputStream( b1s );
     }
     coral::Attribute const & dipTimeAttribute =  bunchConf1Cursor.currentRow()[ std::string( "DIPTIME" ) ];
     coral::Attribute const & bunchConf1Attribute =  bunchConf1Cursor.currentRow()[ std::string( "BUCKET" ) ];
     if( !dipTimeAttribute.isNull() and !bunchConf1Attribute.isNull() ){
       cond::Time_t dipTime = cond::time::from_boost( dipTimeAttribute.data<coral::TimeStamp>().time() );
       // assuming only one sample has been selected...
       unsigned short slot = ( bunchConf1Attribute.data<unsigned short>() - 1 ) / 10 + 1;
       if( lumiSectionTime == 0 or lumiSectionTime == dipTime){
	 bunchConfiguration1[slot] = true;	   
       } else break;
       lumiSectionTime = dipTime;
     }
   }
   if( ret ){
     m_fillPayload->setBunchBitsetForBeam1(bunchConfiguration1);
   }
   
   //execute query for Beam 2
   std::unique_ptr<coral::IQuery> bunchConf2Query(beamCondSchema.newQuery());
   bunchConf2Query->addToTableList( std::string( "LHC_CIRCBUNCHCONFIG_BEAM2" ), std::string( "BEAMCONF\", TABLE( BEAMCONF.VALUE ) \"BUCKETS" ) );
   bunchConf2Query->addToOutputList( std::string( "BEAMCONF.DIPTIME" ), std::string( "DIPTIME" ) );
   bunchConf2Query->addToOutputList( std::string( "BUCKETS.COLUMN_VALUE" ), std::string( "BUCKET" ) );
   bunchConf2Query->setCondition( conditionStr, bunchConfBindVariables );
   bunchConf2Query->addToOrderList( std::string( "DIPTIME" ) );
   bunchConf2Query->limitReturnedRows( LHCInfo::availableBunchSlots ); //maximum number of filled bunches
   bunchConf2Query->defineOutput( bunchConfOutput );
   coral::ICursor& bunchConf2Cursor = bunchConf2Query->execute();

   std::bitset<LHCInfo::bunchSlots+1> bunchConfiguration2( 0ULL );
   ret = false;
   lumiSectionTime = 0;
   while( bunchConf2Cursor.next() ) {
     if( m_debug ) {
       std::ostringstream b2s;
       bunchConf2Cursor.currentRow().toOutputStream( b2s );
     }
     coral::Attribute const & dipTimeAttribute =  bunchConf2Cursor.currentRow()[ std::string( "DIPTIME" ) ];
     coral::Attribute const & bunchConf2Attribute =  bunchConf2Cursor.currentRow()[ std::string( "BUCKET" ) ];
     if( !dipTimeAttribute.isNull() and !bunchConf2Attribute.isNull() ){
       ret = true;
       cond::Time_t dipTime = cond::time::from_boost( dipTimeAttribute.data<coral::TimeStamp>().time() );
       // assuming only one sample has been selected...                                                                                                                     
       unsigned short slot = ( bunchConf2Attribute.data<unsigned short>() - 1 ) / 10 + 1;
       if( lumiSectionTime == 0 or lumiSectionTime == dipTime){
         bunchConfiguration2[slot] = true;
       } else break;
       lumiSectionTime = dipTime;
     }
   }
   if(ret){
     m_fillPayload->setBunchBitsetForBeam2( bunchConfiguration2 );
   }
   //execute query for lumiPerBX
   std::unique_ptr<coral::IQuery> lumiDataQuery(beamCondSchema.newQuery());
   lumiDataQuery->addToTableList( std::string( "CMS_LHC_LUMIPERBUNCH" ), std::string( "LUMIPERBUNCH\", TABLE( LUMIPERBUNCH.LUMI_BUNCHINST ) \"VALUE" ) );
   lumiDataQuery->addToOutputList( std::string( "LUMIPERBUNCH.DIPTIME" ), std::string( "DIPTIME" ) );
   lumiDataQuery->addToOutputList( std::string( "VALUE.COLUMN_VALUE" ), std::string( "LUMI_BUNCH" ) );
   coral::AttributeList lumiDataBindVariables;
   lumiDataBindVariables.extend<coral::TimeStamp>( std::string( "beginFillTime" ) );
   lumiDataBindVariables.extend<coral::TimeStamp>( std::string( "endFillTime" ) );
   lumiDataBindVariables[ std::string( "beginFillTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp(beginFillTime);
   lumiDataBindVariables[ std::string( "endFillTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp(endFillTime);
   conditionStr = std::string( "DIPTIME BETWEEN :beginFillTime AND :endFillTime" );
   lumiDataQuery->setCondition( conditionStr, lumiDataBindVariables );
   lumiDataQuery->addToOrderList( std::string( "DIPTIME" ) );
   lumiDataQuery->limitReturnedRows(3564); //Maximum number of bunches.
   //define query output
   coral::AttributeList lumiDataOutput;
   lumiDataOutput.extend<coral::TimeStamp>( std::string( "DIPTIME" ) );
   lumiDataOutput.extend<float>( std::string( "LUMI_BUNCH" ) );
   lumiDataQuery->defineOutput( lumiDataOutput );
   //execute the query
   coral::ICursor& lumiDataCursor = lumiDataQuery->execute();

   std::vector<float> lumiPerBX;
   ret = false;
   lumiSectionTime = 0;
   while( lumiDataCursor.next() ) {
     if( m_debug ) {
       std::ostringstream lpBX;
       lumiDataCursor.currentRow().toOutputStream( lpBX );
     }
     coral::Attribute const & dipTimeAttribute =  lumiDataCursor.currentRow()[ std::string( "DIPTIME" ) ];
     coral::Attribute const & lumiBunchAttribute =  lumiDataCursor.currentRow()[ std::string( "LUMI_BUNCH" ) ];
     if( !dipTimeAttribute.isNull() and !lumiBunchAttribute.isNull() ){
       ret = true;
       cond::Time_t dipTime = cond::time::from_boost( dipTimeAttribute.data<coral::TimeStamp>().time() );
       // assuming only one sample has been selected...
       float lumi_b = lumiBunchAttribute.data<float>();
       if( lumiSectionTime == 0 or lumiSectionTime == dipTime){
	 if( lumi_b != 0.00 ) lumiPerBX.push_back( lumi_b );
       } else break;
       lumiSectionTime = dipTime;
     }
   }
   if( ret){
     m_fillPayload->setLumiPerBX( lumiPerBX );
   } 
   return ret;
 }

 bool LHCInfoPopConSourceHandler::getCTTPSData( cond::persistency::Session& session, 
					      const boost::posix_time::ptime& beginFillTime, 
					      const boost::posix_time::ptime& endFillTime  ){
   //run the fifth query against the CTPPS schema
   //Initializing the CMS_CTP_CTPPS_COND schema.
   coral::ISchema& CTPPS = session.coralSession().schema("CMS_CTP_CTPPS_COND");
   //execute query for CTPPS Data
   std::unique_ptr<coral::IQuery> CTPPSDataQuery( CTPPS.newQuery() );
   //FROM clause
   CTPPSDataQuery->addToTableList( std::string( "CTPPS_LHC_MACHINE_PARAMS" ) );
   //SELECT clause
   CTPPSDataQuery->addToOutputList( std::string( "DIP_UPDATE_TIME" ) );
   CTPPSDataQuery->addToOutputList( std::string( "LHC_STATE" ) );
   CTPPSDataQuery->addToOutputList( std::string( "LHC_COMMENT" ) );
   CTPPSDataQuery->addToOutputList( std::string( "CTPPS_STATUS" ) );
   CTPPSDataQuery->addToOutputList( std::string( "LUMI_SECTION" ) );
   CTPPSDataQuery->addToOutputList( std::string( "XING_ANGLE_URAD" ) );
   CTPPSDataQuery->addToOutputList( std::string( "BETA_STAR_CMS" ) );
   //WHERE CLAUSE
   coral::AttributeList CTPPSDataBindVariables;
   CTPPSDataBindVariables.extend<coral::TimeStamp>( std::string( "beginFillTime" ) );
   CTPPSDataBindVariables.extend<coral::TimeStamp>( std::string( "endFillTime" ) );
   CTPPSDataBindVariables[ std::string( "beginFillTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp( beginFillTime );
   CTPPSDataBindVariables[ std::string( "endFillTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp( endFillTime );
   std::string conditionStr = std::string( "DIP_UPDATE_TIME>= :beginFillTime and DIP_UPDATE_TIME< :endFillTime" );
   CTPPSDataQuery->setCondition( conditionStr, CTPPSDataBindVariables );
    //ORDER BY clause
   CTPPSDataQuery->addToOrderList( std::string( "DIP_UPDATE_TIME" ) ); 
   //define query output
   coral::AttributeList CTPPSDataOutput;
   CTPPSDataOutput.extend<coral::TimeStamp>( std::string( "DIP_UPDATE_TIME" ) );
   CTPPSDataOutput.extend<std::string>( std::string( "LHC_STATE" ) );
   CTPPSDataOutput.extend<std::string>( std::string( "LHC_COMMENT" ) );
   CTPPSDataOutput.extend<std::string>( std::string( "CTPPS_STATUS" ) );
   CTPPSDataOutput.extend<int>( std::string( "LUMI_SECTION" ) );
   CTPPSDataOutput.extend<float>( std::string( "XING_ANGLE_URAD" ) );
   CTPPSDataOutput.extend<float>( std::string( "BETA_STAR_CMS" ) );
   CTPPSDataQuery->defineOutput( CTPPSDataOutput );
   //execute the query
   coral::ICursor& CTPPSDataCursor = CTPPSDataQuery->execute();
   cond::Time_t dipTime = 0;
   std::string lhcState = "", lhcComment = "", ctppsStatus = "";
   unsigned int lumiSection = 0;
   float crossingAngle = 0., betastar = 0.; 
   
   bool ret = false;
   LHCInfoImpl::LumiSectionFilter filter( m_tmpBuffer );
   while( CTPPSDataCursor.next() ) {
     if( m_debug ) {
       std::ostringstream CTPPS;
       CTPPSDataCursor.currentRow().toOutputStream( CTPPS );
     }
     coral::Attribute const & dipTimeAttribute = CTPPSDataCursor.currentRow()[ std::string( "DIP_UPDATE_TIME" ) ];
     if( !dipTimeAttribute.isNull() ) {
       dipTime = cond::time::from_boost( dipTimeAttribute.data<coral::TimeStamp>().time() );
       if( filter.process( dipTime ) ){
         ret = true;
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
	 coral::Attribute const & crossingAngleAttribute = CTPPSDataCursor.currentRow()[ std::string( "XING_ANGLE_URAD" ) ];
	 if( !crossingAngleAttribute.isNull() ) {
	   crossingAngle = crossingAngleAttribute.data<float>();
	 }
	 coral::Attribute const & betaStarAttribute = CTPPSDataCursor.currentRow()[ std::string( "BETA_STAR_CMS" ) ];
	 if( !betaStarAttribute.isNull() ) {
	   betastar = betaStarAttribute.data<float>();
	 }
         for( auto it = filter.current(); it!=m_tmpBuffer.end(); it++ ){
	   // set the current values to all of the payloads of the lumi section samples after the current since 
	   LHCInfo& payload = *(it->second);
	   payload.setCrossingAngle( crossingAngle );
	   payload.setBetaStar( betastar );
	   payload.setLhcState( lhcState );
	   payload.setLhcComment( lhcComment );
	   payload.setCtppsStatus( ctppsStatus );
	   payload.setLumiSection( lumiSection );
	 }
       }
     }
   }
   return ret;
 }

namespace LHCInfoImpl {
  static const std::map<std::string, int> vecMap = {{"Beam1/beamPhaseMean",1},{"Beam2/beamPhaseMean",2},{"Beam1/cavPhaseMean",3},{"Beam2/cavPhaseMean",4}};
  void setElementData( cond::Time_t since, const std::string& dipVal,
		       unsigned int elementNr, float value, 
		       LHCInfo& payload, std::set<cond::Time_t>& initList ){
    if( initList.find(since)==initList.end() ){
      payload.beam1VC().resize( LHCInfo::bunchSlots ,0.);
      payload.beam2VC().resize( LHCInfo::bunchSlots ,0.);
      payload.beam1RF().resize( LHCInfo::bunchSlots,0.);
      payload.beam2RF().resize( LHCInfo::bunchSlots,0.);
      initList.insert(since);
    }
    // set the current values to all of the payloads of the lumi section samples after the current since 
    if( elementNr < LHCInfo::bunchSlots ){
      switch( vecMap.at(dipVal) ){
      case 1:
	payload.beam1VC()[elementNr]=value; 
	break;
      case 2: 
	payload.beam2VC()[elementNr]=value;
	break;
      case 3:
	payload.beam1RF()[elementNr]=value;
	break;
      case 4:
	payload.beam2RF()[elementNr]=value;
	break;
      default:
	break;
      }
    }
  }
}

bool LHCInfoPopConSourceHandler::getEcalData(  cond::persistency::Session& session, 
					       const boost::posix_time::ptime& lowerTime, 
					       const boost::posix_time::ptime& upperTime,
					       bool update ){
  //run the sixth query against the CMS_DCS_ENV_PVSS_COND schema
  //Initializing the CMS_DCS_ENV_PVSS_COND schema.
  coral::ISchema& ECAL = session.nominalSchema();
  //start the transaction against the fill logging schema
  //execute query for ECAL Data
  std::unique_ptr<coral::IQuery> ECALDataQuery( ECAL.newQuery() );
  //FROM clause
  ECALDataQuery->addToTableList( std::string( "BEAM_PHASE" ) );
  //SELECT clause 
  ECALDataQuery->addToOutputList( std::string( "CHANGE_DATE" ) );
  ECALDataQuery->addToOutputList( std::string( "DIP_value" ) );
  ECALDataQuery->addToOutputList( std::string( "element_nr" ) );
  ECALDataQuery->addToOutputList( std::string( "VALUE_NUMBER" ) );
  //WHERE CLAUSE
  coral::AttributeList ECALDataBindVariables;
  ECALDataBindVariables.extend<coral::TimeStamp>( std::string( "lowerTime" ) );
  ECALDataBindVariables.extend<coral::TimeStamp>( std::string( "upperTime" ) );
  ECALDataBindVariables[ std::string( "lowerTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp( lowerTime );
  ECALDataBindVariables[ std::string( "upperTime" ) ].data<coral::TimeStamp>() = coral::TimeStamp( upperTime );
  std::string conditionStr = 
    std::string( "(DIP_value LIKE '%beamPhaseMean%' OR DIP_value LIKE '%cavPhaseMean%') AND CHANGE_DATE >= :lowerTime AND CHANGE_DATE < :upperTime" );
  
  ECALDataQuery->setCondition( conditionStr, ECALDataBindVariables );
  //ORDER BY clause
  ECALDataQuery->addToOrderList( std::string( "CHANGE_DATE" ) );
  ECALDataQuery->addToOrderList( std::string( "DIP_value" ) );
  ECALDataQuery->addToOrderList( std::string( "element_nr" ) );
  //define query output
  coral::AttributeList ECALDataOutput;
  ECALDataOutput.extend<coral::TimeStamp>( std::string( "CHANGE_DATE" ) );
  ECALDataOutput.extend<std::string>( std::string( "DIP_value" ) );
  ECALDataOutput.extend<unsigned int>( std::string( "element_nr" ) );
  ECALDataOutput.extend<float>( std::string( "VALUE_NUMBER" ) );
  //ECALDataQuery->limitReturnedRows( 14256 ); //3564 entries per vector.
  ECALDataQuery->defineOutput( ECALDataOutput );
  //execute the query
  coral::ICursor& ECALDataCursor = ECALDataQuery->execute();
  cond::Time_t changeTime = 0;
  cond::Time_t firstTime = 0;
  std::string dipVal = "";
  unsigned int elementNr = 0;
  float value = 0.;
  std::set<cond::Time_t> initializedVectors;
  LHCInfoImpl::LumiSectionFilter filter( m_tmpBuffer );
  bool ret = false;
  if(m_prevPayload.get()){
    for(auto& lumiSlot: m_tmpBuffer){
      lumiSlot.second->setBeam1VC( m_prevPayload->beam1VC() );
      lumiSlot.second->setBeam2VC( m_prevPayload->beam2VC() );
      lumiSlot.second->setBeam1RF( m_prevPayload->beam1RF() );
      lumiSlot.second->setBeam2RF( m_prevPayload->beam2RF() );
    }
  }
  std::map<cond::Time_t,cond::Time_t> iovMap;
  cond::Time_t lowerLumi = m_tmpBuffer.front().first;
  while( ECALDataCursor.next() ) {
    if( m_debug ) {
      std::ostringstream ECAL;
      ECALDataCursor.currentRow().toOutputStream( ECAL );
    }
    coral::Attribute const & changeDateAttribute = ECALDataCursor.currentRow()[ std::string( "CHANGE_DATE" ) ];
    if( !changeDateAttribute.isNull() ) {
      ret = true;
      boost::posix_time::ptime  chTime = changeDateAttribute.data<coral::TimeStamp>().time();
      // move the first IOV found to the start of the fill interval selected
      if( changeTime == 0 ) {
	firstTime = cond::time::from_boost( chTime );
      }
      changeTime = cond::time::from_boost( chTime );      
      cond::Time_t iovTime = changeTime;
      if( !update and changeTime == firstTime ) iovTime = lowerLumi;
      coral::Attribute const & dipValAttribute = ECALDataCursor.currentRow()[ std::string( "DIP_value" ) ];
      coral::Attribute const & valueNumberAttribute = ECALDataCursor.currentRow()[ std::string( "VALUE_NUMBER" ) ];
      coral::Attribute const & elementNrAttribute = ECALDataCursor.currentRow()[ std::string( "element_nr" ) ];
      if( !dipValAttribute.isNull() and !valueNumberAttribute.isNull() ) {
	dipVal = dipValAttribute.data<std::string>();
	elementNr = elementNrAttribute.data<unsigned int>();
	value = valueNumberAttribute.data<float>();
        if( isnan( value ) ) value = 0.;
	if( filter.process( iovTime ) ){    
          iovMap.insert(std::make_pair( changeTime, filter.current()->first ) );
	  for( auto it = filter.current(); it!=m_tmpBuffer.end(); it++ ){
	    LHCInfo& payload = *(it->second);
	    LHCInfoImpl::setElementData(it->first, dipVal, elementNr, value, payload, initializedVectors );
	  }
	}
	  //} 
      }
    }
  }
  if( m_debug ){
    for( auto& im: iovMap ){
      edm::LogInfo(m_name) <<"Found iov="<<im.first<<" ("<<cond::time::to_boost(im.first)<<" ) moved to "<<im.second<<" ( "<<cond::time::to_boost(im.second)<<" )";
    }
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
    auto newPayload = std::make_shared<LHCInfo>();
    m_to_transfer.push_back( std::make_pair( newPayload.get(), iov ) );
    m_payloadBuffer.push_back(newPayload);
    m_prevPayload = newPayload;
  }
}

namespace LHCInfoImpl {
  bool comparePayloads( const LHCInfo& rhs, const LHCInfo& lhs ){
    if( rhs.fillNumber() != lhs.fillNumber() ) return false;
    if( rhs.delivLumi()  != lhs.delivLumi() ) return false;
    if( rhs.recLumi()  != lhs.recLumi() ) return false;
    if( rhs.instLumi()  != lhs.instLumi() ) return false;
    if( rhs.instLumiError()  != lhs.instLumiError() ) return false;
    if( rhs.crossingAngle() != rhs.crossingAngle() ) return false;
    if( rhs.betaStar() != rhs.betaStar() ) return false;
    if( rhs.lhcState() != rhs.lhcState() ) return false;
    if( rhs.lhcComment() != rhs.lhcComment() ) return false;
    if( rhs.ctppsStatus() != rhs.ctppsStatus() ) return false;
    return true;
  }

  size_t transferPayloads( const std::vector<std::pair<cond::Time_t,std::shared_ptr<LHCInfo> > >& buffer, 
			   std::vector<std::shared_ptr<LHCInfo> >& payloadBuffer,
			   std::vector<std::pair<LHCInfo*,cond::Time_t> >& vecToTransfer,
			   std::shared_ptr<LHCInfo>& prevPayload ){
    size_t niovs = 0;
    for( auto& iov: buffer ){
      bool add = false;
      LHCInfo& payload = *iov.second;
      cond::Time_t since = iov.first;
      if( vecToTransfer.empty() ){
	add = true;
      } else {
	LHCInfo& lastAdded = *vecToTransfer.back().first;
	if( !comparePayloads( lastAdded,payload ) ) {
	  add = true;
	}
      }
      if( add ) {
        niovs++;
	vecToTransfer.push_back( std::make_pair( &payload, since  ) );
        payloadBuffer.push_back( iov.second );
        prevPayload = iov.second;
      } 
    }
    return niovs;
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

  cond::Time_t lastSince = tagInfo().lastInterval.first; 
  if( lastSince == 0 ){
    // for a new or empty tag, an empty payload should be added on top with since=1
    addEmptyPayload( 1 );
  } else {
    edm::LogInfo( m_name ) << "The last Iov in tag " << tagInfo().name 
			   << " valid since " << lastSince
			   << "from " << m_name << "::getNewObjects";
  }

  boost::posix_time::ptime executionTime = boost::posix_time::second_clock::local_time();
  cond::Time_t targetSince = 0;  
  cond::Time_t endIov = cond::time::from_boost( executionTime );  
  if( !m_startTime.is_not_a_date_time() ){
    targetSince = cond::time::from_boost(m_startTime);
  }
  if( lastSince > targetSince ) targetSince = lastSince;

  edm::LogInfo(m_name) <<"Starting sampling at "<<boost::posix_time::to_simple_string(cond::time::to_boost(targetSince));
  
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
  // fetch last payload when available
  if( !tagInfo().lastPayloadToken.empty() ){
    cond::persistency::Session session3 = dbSession();
    session3.transaction().start(true);
    m_prevPayload = session3.fetchPayload<LHCInfo>( tagInfo().lastPayloadToken );
    session3.transaction().commit();
  }

  bool iovAdded = false;
  while( true ){
    if( targetSince >= endIov ){
      edm::LogInfo( m_name ) <<"Sampling ended at the time "<<boost::posix_time::to_simple_string(cond::time::to_boost( endIov ));
      break;
    }
    bool updateEcal=false;
    boost::posix_time::ptime targetTime = cond::time::to_boost( targetSince );
    boost::posix_time::ptime startSampleTime; 
    boost::posix_time::ptime endSampleTime;
    if( !m_endFill and m_prevPayload->fillNumber() and m_prevPayload->endTime()==0ULL){
      // execute the query for the current fill 
      session.transaction().start(true);
      edm::LogInfo( m_name ) <<"Searching started fill #"<<m_prevPayload->fillNumber();
      bool foundFill = getFillData( session, m_prevPayload->fillNumber() );
      session.transaction().commit();
      if(!foundFill ){
	edm::LogError( m_name )<<"Could not find fill #"<<m_prevPayload->fillNumber();
	break;
      }
      updateEcal = true;
      startSampleTime = cond::time::to_boost(lastSince);
    } else {
      session.transaction().start(true);
      edm::LogInfo( m_name ) <<"Searching new fill after "<<boost::posix_time::to_simple_string(targetTime);
      bool foundFill = getNextFillData( session, targetTime, m_endFill );
      session.transaction().commit();
      if ( !foundFill ){
	edm::LogInfo( m_name )<<"No fill found - END of job.";
	if( iovAdded ) addEmptyPayload( targetSince );
	break;
      }
      startSampleTime = cond::time::to_boost(m_fillPayload->createTime());
    }
    cond::Time_t startFillTime = m_fillPayload->createTime();
    cond::Time_t endFillTime = m_fillPayload->endTime();
    unsigned short lhcFill = m_fillPayload->fillNumber();
    if( endFillTime == 0ULL ){
      edm::LogInfo( m_name ) <<"Found ongoing fill "<<lhcFill<<" created at "<<cond::time::to_boost(startFillTime);
      endSampleTime = executionTime;
      targetSince = endIov;
    } else {
      edm::LogInfo( m_name ) <<"Found fill "<<lhcFill<<" created at "<<cond::time::to_boost(startFillTime)<<" ending at "<<cond::time::to_boost(endFillTime);
      endSampleTime = cond::time::to_boost(endFillTime);
      targetSince = endFillTime;
    }

    session.transaction().start(true);
    getDipData( session, startSampleTime, endSampleTime );
    size_t nlumi = getLumiData( session, startSampleTime, endSampleTime );
    edm::LogInfo( m_name ) <<"Found "<<nlumi<<" lumisections during the fill "<<lhcFill;
    boost::posix_time::ptime flumiStart = cond::time::to_boost(m_tmpBuffer.front().first);
    boost::posix_time::ptime flumiStop = cond::time::to_boost(m_tmpBuffer.back().first);
    edm::LogInfo( m_name ) <<"First lumi starts at "<<flumiStart<<" last lumi starts at "<<flumiStop;
    getCTTPSData( session, startSampleTime, endSampleTime );
    session.transaction().commit();
    session2.transaction().start(true);
    getEcalData( session2, startSampleTime, endSampleTime, updateEcal );
    session2.transaction().commit();
    // 
    size_t niovs = LHCInfoImpl::transferPayloads( m_tmpBuffer, m_payloadBuffer, m_to_transfer, m_prevPayload );
    edm::LogInfo( m_name ) <<"Added "<<niovs<<" iovs within the Fill time";
    m_tmpBuffer.clear();
    iovAdded = true;
    //if(m_prevPayload->fillNumber() and m_prevPayload->endTime()!=0ULL) addEmptyPayload( m_fillPayload->endTime() );
    if(m_prevPayload->fillNumber() and m_fillPayload->endTime()!=0ULL) addEmptyPayload( m_fillPayload->endTime() );
  }
}

std::string LHCInfoPopConSourceHandler::id() const { 
  return m_name;
}

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondTools/RunInfo/interface/FillInfoPopConSourceHandler.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
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

FillInfoPopConSourceHandler::FillInfoPopConSourceHandler( edm::ParameterSet const & pset ):
  m_debug( pset.getUntrackedParameter<bool>( "debug", false ) )
  ,m_fill( (unsigned short)pset.getUntrackedParameter<unsigned int>( "fill", 1 ) )
  ,m_name( pset.getUntrackedParameter<std::string>( "name", "FillInfoPopConSourceHandler" ) )
  ,m_connectionString(pset.getUntrackedParameter<std::string>("connectionString",""))
  ,m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath","")) {}

FillInfoPopConSourceHandler::~FillInfoPopConSourceHandler() {}

void FillInfoPopConSourceHandler::getNewObjects() {
  
  FillInfo* fillInfo = new FillInfo( m_fill ); 
  
  // transfer fake fill for 1 to since for the first time
  if ( tagInfo().size == 0 && m_fill != 1 ) {
    edm::LogInfo("FillInfoPopConSourceHandler") << "New tag "<< tagInfo().name << ": inserting first fake object; from " << m_name << "::getNewObjects" << std::endl;
    m_to_transfer.push_back( std::make_pair( new FillInfo(),1 ) );
  } else {
    //checks what is already inside of database
    edm::LogInfo("FillInfoPopConSourceHandler") << "got info for tag " << tagInfo().name 
						<< ", IOVSequence token " << tagInfo().token
						<< ": size " << tagInfo().size 
						<< ", last object valid since " << tagInfo().lastInterval.first 
						<< " ( "<< boost::posix_time::to_iso_extended_string( cond::time::to_boost( tagInfo().lastInterval.first ) )
						<< " ); from " << m_name << "::getNewObjects"  
						<< std::endl;
  }
  //reading from omds
  cond::DbConnection dbConnection;
  if(m_debug) {
    dbConnection.configuration().setMessageLevel( coral::Debug );
  } else {
    dbConnection.configuration().setMessageLevel( coral::Error );
  }
  dbConnection.configuration().setPoolAutomaticCleanUp( false );
  dbConnection.configuration().setConnectionTimeOut( 0 );
  dbConnection.configuration().setAuthenticationPath( m_authpath );
  dbConnection.configure();
  cond::DbSession session = dbConnection.createSession();
  session.open( m_connectionString, true );
  coral::ISchema& runTimeLoggerSchema = session.nominalSchema();
  session.transaction().start(true);
  std::unique_ptr<coral::IQuery> fillDataQuery( runTimeLoggerSchema.newQuery() );
  fillDataQuery->addToTableList( std::string( "RUNTIME_SUMMARY" ) );
  fillDataQuery->addToOutputList( "NBUNCHESBEAM1" );
  fillDataQuery->addToOutputList( "NBUNCHESBEAM2" );
  fillDataQuery->addToOutputList( "NCOLLIDINGBUNCHES" );
  fillDataQuery->addToOutputList( "NTARGETBUNCHES" );
  fillDataQuery->addToOutputList( "RUNTIME_TYPE_ID" );
  fillDataQuery->addToOutputList( "PARTY1" );
  fillDataQuery->addToOutputList( "PARTY2" );
  fillDataQuery->addToOutputList( "CROSSINGANGLE" );
  fillDataQuery->addToOutputList( "BETASTAR" );
  fillDataQuery->addToOutputList( "INTENSITYBEAM1" );
  fillDataQuery->addToOutputList( "INTENSITYBEAM2" );
  fillDataQuery->addToOutputList( "ENERGY" );
  fillDataQuery->addToOutputList( "CREATETIME" );
  fillDataQuery->addToOutputList( "BEGINTIME" );
  fillDataQuery->addToOutputList( "ENDTIME" );
  fillDataQuery->addToOutputList( "INJECTIONSCHEME" );
  coral::AttributeList fillDataBindVariables;
  fillDataBindVariables.extend( "fillNumber", typeid( unsigned short ) );
  fillDataBindVariables[ "fillNumber" ].data<unsigned short>() = m_fill;
  std::string conditionStr( "LHCFILL=:fillNumber" );
  fillDataQuery->setCondition( conditionStr, fillDataBindVariables );
  coral::AttributeList fillDataOutput;
  fillDataOutput.extend<unsigned short>( "NBUNCHESBEAM1" );
  fillDataOutput.extend<unsigned short>( "NBUNCHESBEAM2" );
  fillDataOutput.extend<unsigned short>( "NCOLLIDINGBUNCHES" );
  fillDataOutput.extend<unsigned short>( "NTARGETBUNCHES" );
  fillDataOutput.extend<int>( "RUNTIME_TYPE_ID" );
  fillDataOutput.extend<int>( "PARTY1" );
  fillDataOutput.extend<int>( "PARTY2" );
  fillDataOutput.extend<float>( "CROSSINGANGLE" );
  fillDataOutput.extend<float>( "BETASTAR" );
  fillDataOutput.extend<float>( "INTENSITYBEAM1" );
  fillDataOutput.extend<float>( "INTENSITYBEAM2" );
  fillDataOutput.extend<float>( "ENERGY" );
  fillDataOutput.extend<coral::TimeStamp>( "CREATETIME" );
  fillDataOutput.extend<coral::TimeStamp>( "BEGINTIME" );
  fillDataOutput.extend<coral::TimeStamp>( "ENDTIME" );
  fillDataOutput.extend<std::string>( "INJECTIONSCHEME" );
  fillDataQuery->defineOutput( fillDataOutput );
  coral::ICursor& fillDataCursor = fillDataQuery->execute();
  unsigned short bunches1 = 0, bunches2 = 0, collidingBunches = 0, targetBunches = 0;
  FillInfo::FillTypeId fillType = FillInfo::UNKNOWN;
  FillInfo::ParticleTypeId particleType1 = FillInfo::NONE , particleType2 = FillInfo::NONE;
  float crossingAngle = 0., betastar = 0., intensityBeam1 = 0., intensityBeam2 = 0., energy = 0.;
  coral::TimeStamp stableBeamStartTimeStamp, stableBeamEndTimeStamp;
  cond::Time_t creationTime = 0ULL, stableBeamStartTime = 0ULL, stableBeamEndTime = 0ULL;
  std::string injectionScheme("None");
  while( fillDataCursor.next() ) {
    //fillDataCursor.currentRow().toOutputStream( std::cout ) << std::endl;
    coral::Attribute const & bunches1Attribute = fillDataCursor.currentRow()[ "NBUNCHESBEAM1" ];
    if( bunches1Attribute.isNull() ) {
      bunches1 = 0;
    } else {
      bunches1 = bunches1Attribute.data<unsigned short>();
    }
    coral::Attribute const & bunches2Attribute = fillDataCursor.currentRow()[ "NBUNCHESBEAM2" ];
    if( bunches2Attribute.isNull() ) {
      bunches2 = 0;
    } else {
      bunches2 = bunches2Attribute.data<unsigned short>();
    }
    coral::Attribute const & collidingBunchesAttribute = fillDataCursor.currentRow()[ "NCOLLIDINGBUNCHES" ];
    if( collidingBunchesAttribute.isNull() ) {
      collidingBunches = 0;
    } else {
      collidingBunches = collidingBunchesAttribute.data<unsigned short>();
    }
    coral::Attribute const & targetBunchesAttribute = fillDataCursor.currentRow()[ "NTARGETBUNCHES" ];
    if( targetBunchesAttribute.isNull() ) {
      targetBunches = 0;
    } else {
      targetBunches = targetBunchesAttribute.data<unsigned short>();
    }
    //RUNTIME_TYPE_ID IS NOT NULL
    fillType = static_cast<FillInfo::FillTypeId>( fillDataCursor.currentRow()[ "RUNTIME_TYPE_ID" ].data<int>() );
    coral::Attribute const & particleType1Attribute = fillDataCursor.currentRow()[ "PARTY1" ];
    if( particleType1Attribute.isNull() ) {
      particleType1 = FillInfo::NONE;
    } else {
      particleType1 = static_cast<FillInfo::ParticleTypeId>( particleType1Attribute.data<int>() );
    }
    coral::Attribute const & particleType2Attribute = fillDataCursor.currentRow()[ "PARTY2" ];
    if( particleType2Attribute.isNull() ) {
      particleType2 = FillInfo::NONE;
    } else {
      particleType2 = static_cast<FillInfo::ParticleTypeId>( particleType2Attribute.data<int>() );
    }
    coral::Attribute const & crossingAngleAttribute = fillDataCursor.currentRow()[ "CROSSINGANGLE" ];
    if( crossingAngleAttribute.isNull() ) {
      crossingAngle = 0.;
    } else {
      crossingAngle = crossingAngleAttribute.data<float>();
    }
    coral::Attribute const & betastarAttribute = fillDataCursor.currentRow()[ "BETASTAR" ];
    if( betastarAttribute.isNull() ) {
      betastar = 0.;
    } else {
      betastar = betastarAttribute.data<float>();
    }
    coral::Attribute const & intensityBeam1Attribute = fillDataCursor.currentRow()[ "INTENSITYBEAM1" ];
    if( intensityBeam1Attribute.isNull() ) {
      intensityBeam1 = 0.;
    } else {
      intensityBeam1 = intensityBeam1Attribute.data<float>();
    }
    coral::Attribute const & intensityBeam2Attribute = fillDataCursor.currentRow()[ "INTENSITYBEAM2" ];
    if( intensityBeam2Attribute.isNull() ) {
      intensityBeam2 = 0.;
    } else {
      intensityBeam2 = intensityBeam2Attribute.data<float>();
    }
    coral::Attribute const & energyAttribute = fillDataCursor.currentRow()[ "ENERGY" ];
    if( energyAttribute.isNull() ){
      energy = 0.;
    } else {
      energy = energyAttribute.data<float>();
    }
    //CREATETIME IS NOT NULL
    creationTime = cond::time::from_boost( fillDataCursor.currentRow()[ "CREATETIME" ].data<coral::TimeStamp>().time() );
    coral::Attribute const & stableBeamStartTimeAttribute = fillDataCursor.currentRow()[ "BEGINTIME" ];
    if( stableBeamStartTimeAttribute.isNull() ) {
      stableBeamStartTime = 0;
    } else {
      stableBeamStartTimeStamp = stableBeamStartTimeAttribute.data<coral::TimeStamp>();
      stableBeamStartTime = cond::time::from_boost( stableBeamStartTimeStamp.time() );
    }
    coral::Attribute const & stableBeamEndTimeAttribute = fillDataCursor.currentRow()[ "ENDTIME" ];
    if( stableBeamEndTimeAttribute.isNull() ) {
      stableBeamEndTime = 0;
    } else {
      stableBeamEndTimeStamp = stableBeamEndTimeAttribute.data<coral::TimeStamp>();
      stableBeamEndTime = cond::time::from_boost( stableBeamEndTimeStamp.time() );
    }
    coral::Attribute const & injectionSchemeAttribute = fillDataCursor.currentRow()[ "INJECTIONSCHEME" ];
    if( injectionSchemeAttribute.isNull() ) {
      injectionScheme = std::string( "None" );
    } else {
      injectionScheme = injectionSchemeAttribute.data<std::string>();
    }
  }
  session.transaction().commit();
  
  //fixing an inconsistency in RunTimeLogger: if the fill type is defined, the particle type should reflect it!
  if( fillType != FillInfo::UNKNOWN && ( particleType1 == FillInfo::NONE || particleType2 == FillInfo::NONE ) ) {
    switch( fillType ) {
    case FillInfo::PROTONS :
      particleType1 = FillInfo::PROTON;
      particleType2 = FillInfo::PROTON;
      break;
    case FillInfo::IONS :
      particleType1 = FillInfo::PB82;
      particleType2 = FillInfo::PB82;
      break;
    case FillInfo::UNKNOWN :
    case FillInfo::COSMICS :
    case FillInfo::GAP :
      break;
    }
  }
  
  //if the start time of the fill is 0 (i.e. timestamp null), it never went to stable beams: do not store!
  if( stableBeamStartTime == 0 ) {
    edm::LogWarning("FillInfoPopConSourceHandler") << "NO TRANSFER NEEDED: the fill number " << m_fill
						   << " never went into stable beams"
						   << "; from " << m_name << "::getNewObjects" 
						   << std::endl;
    session.close();
    dbConnection.close();
    return;
  }
  
  //if the end time of the fill is 0 (i.e. timestamp null), it is still ongoing: do not store!
  if( stableBeamEndTime == 0 ) {
    edm::LogWarning("FillInfoPopConSourceHandler") << "NO TRANSFER NEEDED: the fill number " << m_fill
						   << " is still ongoing"
						   << "; from " << m_name << "::getNewObjects" 
						   << std::endl;
    session.close();
    dbConnection.close();
    return;
  }
  
  coral::ISchema& beamCondSchema = session.schema( "CMS_BEAM_COND" );
  session.transaction().start( true );
  //preparing the where clause for both queries
  coral::AttributeList bunchConfBindVariables;
  bunchConfBindVariables.extend<coral::TimeStamp>( "stableBeamStartTimeStamp" );
  bunchConfBindVariables[ "stableBeamStartTimeStamp" ].data<coral::TimeStamp>() = stableBeamStartTimeStamp;
  conditionStr = std::string( "DIPTIME <= :stableBeamStartTimeStamp" );
  //defining the output types for both queries
  coral::AttributeList bunchConfOutput;
  bunchConfOutput.extend<coral::TimeStamp>( "DIPTIME" );
  bunchConfOutput.extend<unsigned short>( "BUCKET" );
  //executing query for Beam 1
  std::unique_ptr<coral::IQuery> bunchConf1Query(beamCondSchema.newQuery());
  bunchConf1Query->addToTableList( "LHC_CIRCBUNCHCONFIG_BEAM1", "BEAMCONF\", TABLE( BEAMCONF.VALUE ) \"BUCKETS" );
  bunchConf1Query->addToOutputList( "BEAMCONF.DIPTIME", "DIPTIME" );
  bunchConf1Query->addToOutputList( "BUCKETS.COLUMN_VALUE", "BUCKET" );
  bunchConf1Query->setCondition( conditionStr, bunchConfBindVariables );
  bunchConf1Query->addToOrderList( "DIPTIME DESC" );
  bunchConf1Query->limitReturnedRows( 2808 ); //maximum number of filled bunches
  bunchConf1Query->defineOutput( bunchConfOutput );
  coral::ICursor& bunchConf1Cursor = bunchConf1Query->execute();
  std::bitset<FillInfo::bunchSlots+1> bunchConfiguration1( 0ULL );
  while( bunchConf1Cursor.next() ) {
    //bunchConf1Cursor.currentRow().toOutputStream( std::cout ) << std::endl;
    if( bunchConf1Cursor.currentRow()[ "BUCKET" ].data<unsigned short>() != 0 ) {
      unsigned short slot = ( bunchConf1Cursor.currentRow()[ "BUCKET" ].data<unsigned short>() - 1 ) / 10 + 1;
      bunchConfiguration1[ slot ] = true;
    }
  }
  //executing query for Beam 2
  std::unique_ptr<coral::IQuery> bunchConf2Query(beamCondSchema.newQuery());
  bunchConf2Query->addToTableList( "LHC_CIRCBUNCHCONFIG_BEAM2", "BEAMCONF\", TABLE( BEAMCONF.VALUE ) \"BUCKETS" );
  bunchConf2Query->addToOutputList( "BEAMCONF.DIPTIME", "DIPTIME" );
  bunchConf2Query->addToOutputList( "BUCKETS.COLUMN_VALUE", "BUCKET" );
  bunchConf2Query->setCondition( conditionStr, bunchConfBindVariables );
  bunchConf2Query->addToOrderList( "DIPTIME DESC" );
  bunchConf2Query->limitReturnedRows( 2808 ); //maximum number of filled bunches
  bunchConf2Query->defineOutput( bunchConfOutput );
  coral::ICursor& bunchConf2Cursor = bunchConf2Query->execute();
  std::bitset<FillInfo::bunchSlots+1> bunchConfiguration2( 0ULL );
  while( bunchConf2Cursor.next() ) {
    //bunchConf2Cursor.currentRow().toOutputStream( std::cout ) << std::endl;
    if( bunchConf2Cursor.currentRow()[ "BUCKET" ].data<unsigned short>() != 0 ) {
      unsigned short slot = ( bunchConf2Cursor.currentRow()[ "BUCKET" ].data<unsigned short>() - 1 ) / 10 + 1;
      bunchConfiguration2[ slot ] = true;
    }
  }
  session.transaction().commit();
  session.close();
  dbConnection.close();
  
  //setting values
  fillInfo->setBeamInfo( const_cast<unsigned short const &>( bunches1 )
		       , const_cast<unsigned short const &>( bunches2 )
		       , const_cast<unsigned short const &>( collidingBunches )
		       , const_cast<unsigned short const &>( targetBunches )
		       , const_cast<FillInfo::FillTypeId const &>( fillType )
		       , const_cast<FillInfo::ParticleTypeId const &>( particleType1 )
		       , const_cast<FillInfo::ParticleTypeId const &>( particleType2 )
		       , const_cast<float const &>( crossingAngle )
		       , const_cast<float const &>( betastar )
		       , const_cast<float const &>( intensityBeam1 )
		       , const_cast<float const &>( intensityBeam2 ) 
		       , const_cast<float const &>( energy ) 
		       , const_cast<cond::Time_t const &>( creationTime )
		       , const_cast<cond::Time_t const &>( stableBeamStartTime )
		       , const_cast<cond::Time_t const &>( stableBeamEndTime )
		       , const_cast<std::string const &>( injectionScheme )
		       , const_cast<std::bitset<FillInfo::bunchSlots+1> const &>( bunchConfiguration1 )
		       , const_cast<std::bitset<FillInfo::bunchSlots+1> const &>( bunchConfiguration2 ) );
  
  //retrieving the last payload
  if( tagInfo().size > 0 ) {
    Ref const previousFill = this->lastPayload();
    //checking its content
    edm::LogInfo("FillInfoPopConSourceHandler") << "The last payload in tag " << tagInfo().name 
						<< " valid since " << tagInfo().lastInterval.first
						<< " has token " << tagInfo().lastPayloadToken 
						<< " and values:\n" << *previousFill;
    
    //store dummy fill information if empty fills are found beetween the two last ones in stable beams
    cond::UnpackedTime previousFillEndUnpackedTime = cond::time::unpack( previousFill->endTime() );
    cond::Time_t afterPreviousFillEndTime = cond::time::pack( std::make_pair( previousFillEndUnpackedTime.first, previousFillEndUnpackedTime.second + 1 ) );
    cond::Time_t beforeStableBeamStartTime = cond::time::pack( std::make_pair( cond::time::unpack( stableBeamStartTime ).first, cond::time::unpack( stableBeamStartTime ).second - 1 ) );
    if( afterPreviousFillEndTime < stableBeamStartTime ) {
      edm::LogInfo("FillInfoPopConSourceHandler") << "Entering fake fill between fill number " << previousFill->fillNumber()
						  << " and current fill " << m_fill
						  << ", from " <<  afterPreviousFillEndTime
						  << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( afterPreviousFillEndTime ) )
						  << " ) to " << beforeStableBeamStartTime
						  << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( beforeStableBeamStartTime ) )
						  << " ); from " << m_name << "::getNewObjects" 
						  << std::endl;
      m_to_transfer.push_back( std::make_pair( new FillInfo(), afterPreviousFillEndTime ) );
    } else {
      // no transfer needed: either we are trying to put the same value, or we want to put an older fill
      std::ostringstream es;
      es << "NO TRANSFER NEEDED: trying to insert fill number " << m_fill 
	 << ( ( m_fill < previousFill->fillNumber() ) ? ", which is an older fill than " : ", which is the same fill as " ) 
	 << "the last one in the destination tag " << previousFill->fillNumber(); 
      edm::LogWarning("FillInfoPopConSourceHandler") << es.str()
						     << "; from " << m_name << "::getNewObjects" 
						     << std::endl;
      return;
    }
  }
  m_to_transfer.push_back( std::make_pair( (FillInfo*)fillInfo, stableBeamStartTime ) );
  edm::LogInfo("FillInfoPopConSourceHandler") << "The new payload to be inserted into tag " << tagInfo().name 
					      << " with validity " << stableBeamStartTime 
					      << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( stableBeamStartTime ) )
					      << " ) has values:\n" << *fillInfo
					      << std::endl;
  // adding log information
  std::ostringstream ss;
  ss << " fill = " << m_fill 
     << "; injection scheme: " << injectionScheme
     << "; start time: " 
     << boost::posix_time::to_iso_extended_string( stableBeamStartTimeStamp.time() )
     << "; end time: "
     << boost::posix_time::to_iso_extended_string( stableBeamEndTimeStamp.time() );
  m_userTextLog = ss.str() + ";";
  edm::LogInfo("FillInfoPopConSourceHandler") << "Transferring " << m_to_transfer.size() << " payload(s); from " << m_name << "::getNewObjects" << std::endl;
}

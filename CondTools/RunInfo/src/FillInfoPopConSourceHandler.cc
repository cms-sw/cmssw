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
  ,m_firstFill( (unsigned short)pset.getUntrackedParameter<unsigned int>( "firstFill", 1 ) )
  ,m_lastFill( (unsigned short)pset.getUntrackedParameter<unsigned int>( "lastFill", m_firstFill ) )
  ,m_name( pset.getUntrackedParameter<std::string>( "name", "FillInfoPopConSourceHandler" ) )
  ,m_connectionString(pset.getUntrackedParameter<std::string>("connectionString",""))
  ,m_dipSchema(pset.getUntrackedParameter<std::string>("DIPSchema",""))
  ,m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath","")) {}

FillInfoPopConSourceHandler::~FillInfoPopConSourceHandler() {}

void FillInfoPopConSourceHandler::getNewObjects() {  
  //reference to the last payload in the tag
  Ref previousFill;
  
  //if a new tag is created, transfer fake fill from 1 to the first fill for the first time
  if ( tagInfo().size == 0 ) {
    edm::LogInfo( m_name ) << "New tag "<< tagInfo().name << "; from " << m_name << "::getNewObjects";
  } else {
    //check what is already inside the database
    edm::LogInfo( m_name ) << "got info for tag " << tagInfo().name 
			   << ", IOVSequence token " << tagInfo().token
			   << ": size " << tagInfo().size 
			   << ", last object valid since " << tagInfo().lastInterval.first 
			   << " ( "<< boost::posix_time::to_iso_extended_string( cond::time::to_boost( tagInfo().lastInterval.first ) )
			   << " ); from " << m_name << "::getNewObjects";
    //retrieve the last payload...
    previousFill = this->lastPayload();
    //checking its content
    edm::LogInfo( m_name ) << "The last payload in tag " << tagInfo().name 
			   << " valid since " << tagInfo().lastInterval.first
			   << " has token " << tagInfo().lastPayloadToken 
			   << " and values:\n" << *previousFill
			   << "from " << m_name << "::getNewObjects";
    if( m_firstFill <= previousFill->fillNumber() ) {
      //either we are trying to put the same value, or we want to put an older fill:
      //the first fill will become the previous one plus one 
      std::ostringstream es;
      es << "Trying to insert fill number " << m_firstFill 
	 << ( ( m_firstFill < previousFill->fillNumber() ) ? ", which is an older fill than " : ", which is the same fill as " ) 
	 << "the last one in the destination tag " << previousFill->fillNumber()
	 << ": the first fill to be looked for will become " << previousFill->fillNumber() + 1;
      edm::LogWarning( m_name ) << es.str() << "; from " << m_name << "::getNewObjects";
      m_firstFill = previousFill->fillNumber() + 1;
    }
  }
  
  //if the last fill to be looked for is smaller than the first one send error message and return
  //this check cannot be done before, as we should find which is the first fill to query
  if( m_firstFill > m_lastFill ) {
    edm::LogError( m_name ) << "WRONG CONFIGURATION! The first fill " << m_firstFill
			    << " cannot be larger than the last one " << m_lastFill
			    << " EXITING. from " << m_name << "::getNewObjects";
    return;
  }
  
  //retrieve the data from the relational database source
  cond::DbConnection dbConnection;
  //configure the connection
  if( m_debug ) {
    dbConnection.configuration().setMessageLevel( coral::Debug );
  } else {
    dbConnection.configuration().setMessageLevel( coral::Error );
  }
  dbConnection.configuration().setPoolAutomaticCleanUp( false );
  dbConnection.configuration().setConnectionTimeOut( 0 );
  dbConnection.configuration().setAuthenticationPath( m_authpath );
  dbConnection.configure();
  //create a sessiom
  cond::DbSession session = dbConnection.createSession();
  session.open( m_connectionString, true );
  //run the first query against the schema logging fill information
  coral::ISchema& runTimeLoggerSchema = session.nominalSchema();
  //start the transaction against the fill logging schema
  session.transaction().start(true);
  //prepare the query:
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
  fillDataBindVariables.extend( std::string( "firstFillNumber" ), typeid( unsigned short ) );
  fillDataBindVariables[ std::string( "firstFillNumber" ) ].data<unsigned short>() = m_firstFill;
  fillDataBindVariables.extend( std::string( "lastFillNumber" ), typeid( unsigned short ) );
  fillDataBindVariables[ std::string( "lastFillNumber" ) ].data<unsigned short>() = m_lastFill;
  //by imposing BEGINTIME IS NOT NULL, we remove fills which never went into stable beams,
  //or the most recent one, just declared but not yet in stable beams
  std::string conditionStr( "BEGINTIME IS NOT NULL AND LHCFILL BETWEEN :firstFillNumber AND :lastFillNumber" );
  fillDataQuery->setCondition( conditionStr, fillDataBindVariables );
  //ORDER BY clause
  fillDataQuery->addToOrderList( std::string( "LHCFILL" ) );
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
  //execute the query
  coral::ICursor& fillDataCursor = fillDataQuery->execute();
  //initialize loop variables
  unsigned short previousFillNumber = 1, currentFill = m_firstFill;
  cond::Time_t previousFillEndTime = 0ULL, afterPreviousFillEndTime = 0ULL, beforeStableBeamStartTime = 0ULL;
  if( tagInfo().size > 0 ) {
    previousFillNumber = previousFill->fillNumber();
    previousFillEndTime = previousFill->endTime();
  }
  unsigned short bunches1 = 0, bunches2 = 0, collidingBunches = 0, targetBunches = 0;
  FillInfo::FillTypeId fillType = FillInfo::UNKNOWN;
  FillInfo::ParticleTypeId particleType1 = FillInfo::NONE, particleType2 = FillInfo::NONE;
  float crossingAngle = 0., betastar = 0., intensityBeam1 = 0., intensityBeam2 = 0., energy = 0.;
  coral::TimeStamp stableBeamStartTimeStamp, beamDumpTimeStamp;
  cond::Time_t creationTime = 0ULL, stableBeamStartTime = 0ULL, beamDumpTime = 0ULL;
  std::string injectionScheme( "None" );
  std::ostringstream ss;
  //loop over the cursor where the result of the query were fetched
  while( fillDataCursor.next() ) {
    if( m_debug ) {
      std::ostringstream qs;
      fillDataCursor.currentRow().toOutputStream( qs );
      edm::LogInfo( m_name ) << qs.str() << "\nfrom " << m_name << "::getNewObjects";
    }
    currentFill = fillDataCursor.currentRow()[ std::string( "LHCFILL" ) ].data<unsigned short>();
    coral::Attribute const & bunches1Attribute = fillDataCursor.currentRow()[ std::string( "NBUNCHESBEAM1" ) ];
    if( bunches1Attribute.isNull() ) {
      bunches1 = 0;
    } else {
      bunches1 = bunches1Attribute.data<unsigned short>();
    }
    coral::Attribute const & bunches2Attribute = fillDataCursor.currentRow()[ std::string( "NBUNCHESBEAM2" ) ];
    if( bunches2Attribute.isNull() ) {
      bunches2 = 0;
    } else {
      bunches2 = bunches2Attribute.data<unsigned short>();
    }
    coral::Attribute const & collidingBunchesAttribute = fillDataCursor.currentRow()[ std::string( "NCOLLIDINGBUNCHES" ) ];
    if( collidingBunchesAttribute.isNull() ) {
      collidingBunches = 0;
    } else {
      collidingBunches = collidingBunchesAttribute.data<unsigned short>();
    }
    coral::Attribute const & targetBunchesAttribute = fillDataCursor.currentRow()[ std::string( "NTARGETBUNCHES" ) ];
    if( targetBunchesAttribute.isNull() ) {
      targetBunches = 0;
    } else {
      targetBunches = targetBunchesAttribute.data<unsigned short>();
    }
    //RUNTIME_TYPE_ID IS NOT NULL
    fillType = static_cast<FillInfo::FillTypeId>( fillDataCursor.currentRow()[ std::string( "RUNTIME_TYPE_ID" ) ].data<int>() );
    coral::Attribute const & particleType1Attribute = fillDataCursor.currentRow()[ std::string( "PARTY1" ) ];
    if( particleType1Attribute.isNull() ) {
      particleType1 = FillInfo::NONE;
    } else {
      particleType1 = static_cast<FillInfo::ParticleTypeId>( particleType1Attribute.data<int>() );
    }
    coral::Attribute const & particleType2Attribute = fillDataCursor.currentRow()[ std::string( "PARTY2" ) ];
    if( particleType2Attribute.isNull() ) {
      particleType2 = FillInfo::NONE;
    } else {
      particleType2 = static_cast<FillInfo::ParticleTypeId>( particleType2Attribute.data<int>() );
    }
    coral::Attribute const & crossingAngleAttribute = fillDataCursor.currentRow()[ std::string( "CROSSINGANGLE" ) ];
    if( crossingAngleAttribute.isNull() ) {
      crossingAngle = 0.;
    } else {
      crossingAngle = crossingAngleAttribute.data<float>();
    }
    coral::Attribute const & betastarAttribute = fillDataCursor.currentRow()[ std::string( "BETASTAR" ) ];
    if( betastarAttribute.isNull() ) {
      betastar = 0.;
    } else {
      betastar = betastarAttribute.data<float>();
    }
    coral::Attribute const & intensityBeam1Attribute = fillDataCursor.currentRow()[ std::string( "INTENSITYBEAM1" ) ];
    if( intensityBeam1Attribute.isNull() ) {
      intensityBeam1 = 0.;
    } else {
      intensityBeam1 = intensityBeam1Attribute.data<float>();
    }
    coral::Attribute const & intensityBeam2Attribute = fillDataCursor.currentRow()[ std::string( "INTENSITYBEAM2" ) ];
    if( intensityBeam2Attribute.isNull() ) {
      intensityBeam2 = 0.;
    } else {
      intensityBeam2 = intensityBeam2Attribute.data<float>();
    }
    coral::Attribute const & energyAttribute = fillDataCursor.currentRow()[ std::string( "ENERGY" ) ];
    if( energyAttribute.isNull() ){
      energy = 0.;
    } else {
      energy = energyAttribute.data<float>();
    }
    //CREATETIME IS NOT NULL
    creationTime = cond::time::from_boost( fillDataCursor.currentRow()[ std::string( "CREATETIME" ) ].data<coral::TimeStamp>().time() );
    //BEGINTIME is imposed to be NOT NULL in the WHERE clause
    stableBeamStartTimeStamp = fillDataCursor.currentRow()[ std::string( "BEGINTIME" ) ].data<coral::TimeStamp>();
    stableBeamStartTime = cond::time::from_boost( stableBeamStartTimeStamp.time() );
    coral::Attribute const & beamDumpTimeAttribute = fillDataCursor.currentRow()[ std::string( "ENDTIME" ) ];
    if( beamDumpTimeAttribute.isNull() ) {
      beamDumpTime = 0;
    } else {
      beamDumpTimeStamp = beamDumpTimeAttribute.data<coral::TimeStamp>();
      beamDumpTime = cond::time::from_boost( beamDumpTimeStamp.time() );
    }
    coral::Attribute const & injectionSchemeAttribute = fillDataCursor.currentRow()[ std::string( "INJECTIONSCHEME" ) ];
    if( injectionSchemeAttribute.isNull() ) {
      injectionScheme = std::string( "None" );
    } else {
      injectionScheme = injectionSchemeAttribute.data<std::string>();
    }
    //fix an inconsistency in RunTimeLogger: if the fill type is defined, the particle type should reflect it!
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
    //if the end time of the fill is 0 (i.e. timestamp null), it is still ongoing: do not store!
    if( beamDumpTime == 0 ) {
      edm::LogWarning( m_name ) << "NO TRANSFER NEEDED: the fill number " << currentFill
				<< " is still ongoing"
				<< "; from " << m_name << "::getNewObjects";
      continue;
    }
    //run the second and third query against the schema hosting detailed DIP information
    coral::ISchema& beamCondSchema = session.schema( m_dipSchema );
    //start the transaction against the DIP "deep" database backend schema
    session.transaction().start( true );
    //prepare the WHERE clause for both queries
    coral::AttributeList bunchConfBindVariables;
    bunchConfBindVariables.extend<coral::TimeStamp>( std::string( "stableBeamStartTimeStamp" ) );
    bunchConfBindVariables[ std::string( "stableBeamStartTimeStamp" ) ].data<coral::TimeStamp>() = stableBeamStartTimeStamp;
    conditionStr = std::string( "DIPTIME <= :stableBeamStartTimeStamp" );
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
    bunchConf1Query->limitReturnedRows( FillInfo::availableBunchSlots ); //maximum number of filled bunches
    bunchConf1Query->defineOutput( bunchConfOutput );
    coral::ICursor& bunchConf1Cursor = bunchConf1Query->execute();
    std::bitset<FillInfo::bunchSlots+1> bunchConfiguration1( 0ULL );
    while( bunchConf1Cursor.next() ) {
      if( m_debug ) {
	std::ostringstream b1s;
	fillDataCursor.currentRow().toOutputStream( b1s );
	edm::LogInfo( m_name ) << b1s.str() << "\nfrom " << m_name << "::getNewObjects";
      }
      //bunchConf1Cursor.currentRow().toOutputStream( std::cout ) << std::endl;
      if( bunchConf1Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() != 0 ) {
	unsigned short slot = ( bunchConf1Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() - 1 ) / 10 + 1;
	bunchConfiguration1[ slot ] = true;
      }
    }
    //execute query for Beam 2
    std::unique_ptr<coral::IQuery> bunchConf2Query(beamCondSchema.newQuery());
    bunchConf2Query->addToTableList( std::string( "LHC_CIRCBUNCHCONFIG_BEAM2" ), std::string( "BEAMCONF\", TABLE( BEAMCONF.VALUE ) \"BUCKETS" ) );
    bunchConf2Query->addToOutputList( std::string( "BEAMCONF.DIPTIME" ), std::string( "DIPTIME" ) );
    bunchConf2Query->addToOutputList( std::string( "BUCKETS.COLUMN_VALUE" ), std::string( "BUCKET" ) );
    bunchConf2Query->setCondition( conditionStr, bunchConfBindVariables );
    bunchConf2Query->addToOrderList( std::string( "DIPTIME DESC" ) );
    bunchConf2Query->limitReturnedRows( FillInfo::availableBunchSlots ); //maximum number of filled bunches
    bunchConf2Query->defineOutput( bunchConfOutput );
    coral::ICursor& bunchConf2Cursor = bunchConf2Query->execute();
    std::bitset<FillInfo::bunchSlots+1> bunchConfiguration2( 0ULL );
    while( bunchConf2Cursor.next() ) {
      if( m_debug ) {
	std::ostringstream b2s;
	fillDataCursor.currentRow().toOutputStream( b2s );
	edm::LogInfo( m_name ) << b2s.str() << "\nfrom " << m_name << "::getNewObjects";
      }
      if( bunchConf2Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() != 0 ) {
	unsigned short slot = ( bunchConf2Cursor.currentRow()[ std::string( "BUCKET" ) ].data<unsigned short>() - 1 ) / 10 + 1;
	bunchConfiguration2[ slot ] = true;
      }
    }
    //commit the transaction against the DIP "deep" database backend schema
    session.transaction().commit();
    
    //store dummy fill information if empty fills are found beetween the two last ones in stable beams
    afterPreviousFillEndTime  = cond::time::pack( std::make_pair( cond::time::unpack( previousFillEndTime ).first, cond::time::unpack( previousFillEndTime ).second + 1 ) );
    beforeStableBeamStartTime = cond::time::pack( std::make_pair( cond::time::unpack( stableBeamStartTime ).first, cond::time::unpack( stableBeamStartTime ).second - 1 ) );
    if( afterPreviousFillEndTime < stableBeamStartTime ) {
      edm::LogInfo( m_name ) << "Entering fake fill between fill number " << previousFillNumber
			     << " and current fill number " << currentFill
			     << ", from " <<  afterPreviousFillEndTime
			     << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( afterPreviousFillEndTime ) )
			     << " ) to " << beforeStableBeamStartTime
			     << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( beforeStableBeamStartTime ) )
			     << " ); from " << m_name << "::getNewObjects";
      m_to_transfer.push_back( std::make_pair( new FillInfo(), afterPreviousFillEndTime ) );
    } else {
      //the current fill cannot start before the previous one!
      edm::LogError( m_name ) << "WRONG DATA! In the previous fill number " << previousFillNumber
			      << " beams were dumped at timestamp " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( previousFillEndTime ) )
			      << ", which is not before the timestamp " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( stableBeamStartTime ) )
			      << " when current fill number " << currentFill
			      << " entered stable beams. EXITING. from " << m_name << "::getNewObjects";
      return;
    }
    //construct an instance of FillInfo and set its values
    FillInfo* fillInfo = new FillInfo( currentFill ); 
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
			 , const_cast<cond::Time_t const &>( beamDumpTime )
			 , const_cast<std::string const &>( injectionScheme )
			 , const_cast<std::bitset<FillInfo::bunchSlots+1> const &>( bunchConfiguration1 )
			 , const_cast<std::bitset<FillInfo::bunchSlots+1> const &>( bunchConfiguration2 ) 
			   );
    //store this payload
    m_to_transfer.push_back( std::make_pair( (FillInfo*)fillInfo, stableBeamStartTime ) );
    edm::LogInfo( m_name ) << "The new payload to be inserted into tag " << tagInfo().name 
			   << " with validity " << stableBeamStartTime 
			   << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( stableBeamStartTime ) )
			   << " ) has values:\n" << *fillInfo
			   << "from " << m_name << "::getNewObjects";
    //add log information
    ss << " fill = " << currentFill
       << ";\tinjection scheme: " << injectionScheme
       << ";\tstart time: " 
       << boost::posix_time::to_iso_extended_string( stableBeamStartTimeStamp.time() )
       << ";\tend time: "
       << boost::posix_time::to_iso_extended_string( beamDumpTimeStamp.time() )
       << "." << std::endl;
    //prepare variables for next iteration
    previousFillNumber = currentFill;
    previousFillEndTime = beamDumpTime;
  }
  //commit the transaction against the fill logging schema
  session.transaction().commit();
  //close the session
  session.close();
  //close the connection
  dbConnection.close();
  //store log information
  m_userTextLog = ss.str();
  edm::LogInfo( m_name ) << "Transferring " << m_to_transfer.size() << " payload(s); from " << m_name << "::getNewObjects";
}

std::string FillInfoPopConSourceHandler::id() const { 
  return m_name;
}

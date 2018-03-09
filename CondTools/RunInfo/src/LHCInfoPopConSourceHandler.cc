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
  ,m_firstFill( (unsigned short)pset.getUntrackedParameter<unsigned int>( "firstFill", 1 ) )
  ,m_lastFill( (unsigned short)pset.getUntrackedParameter<unsigned int>( "lastFill", m_firstFill ) )
  ,m_name( pset.getUntrackedParameter<std::string>( "name", "LHCInfoPopConSourceHandler" ) )
  ,m_connectionString(pset.getUntrackedParameter<std::string>("connectionString",""))
  ,m_dipSchema(pset.getUntrackedParameter<std::string>("DIPSchema",""))
  ,m_authpath(pset.getUntrackedParameter<std::string>("authenticationPath","")) {}
//L1: try with different m_dipSchema
//L2: try with different m_name
LHCInfoPopConSourceHandler::~LHCInfoPopConSourceHandler() {}

void LHCInfoPopConSourceHandler::getNewObjects() {
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
  cond::persistency::ConnectionPool connection;
  //configure the connection
  if( m_debug ) {
    connection.setMessageVerbosity( coral::Debug );
  } else {
    connection.setMessageVerbosity( coral::Error );
  }
  connection.setAuthenticationPath( m_authpath );
  connection.configure();
  //create a sessiom
  cond::persistency::Session session = connection.createSession( m_connectionString );
  //run the first query against the schema logging fill information
  coral::ISchema& runTimeLoggerSchema = session.nominalSchema();
  //start the transaction against the fill logging schema
  session.transaction().start(true);
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
  LHCInfo::FillTypeId fillType = LHCInfo::UNKNOWN;
  LHCInfo::ParticleTypeId particleType1 = LHCInfo::NONE, particleType2 = LHCInfo::NONE;
  float crossingAngle = 0., betastar = 0., intensityBeam1 = 0., intensityBeam2 = 0., energy = 0.;
  coral::TimeStamp stableBeamStartTimeStamp, beamDumpTimeStamp;
  cond::Time_t creationTime = 0ULL, stableBeamStartTime = 0ULL, beamDumpTime = 0ULL;
  std::string injectionScheme( "None" );
  std::ostringstream ss;
  
//prepare the query for table 2:
  std::unique_ptr<coral::IQuery> fillDataQuery2( runTimeLoggerSchema.newQuery() );
  //FROM clause
  fillDataQuery2->addToTableList( std::string( "LUMI_SECTIONS" ) );
  //SELECT clause
  fillDataQuery2->addToOutputList( std::string( "MAX(DELIVLUMI)" ) );
  fillDataQuery2->addToOutputList( std::string( "MAX(LIVELUMI)" ) );
  //WHERE clause
  //by imposing BEGINTIME IS NOT NULL, we remove fills which never went into stable beams,
  //or the most recent one, just declared but not yet in stable beams
  conditionStr = "DELIVLUMI IS NOT NULL AND LHCFILL BETWEEN :firstFillNumber AND :lastFillNumber";
  fillDataQuery2->setCondition( conditionStr, fillDataBindVariables );
  //ORDER BY clause
  fillDataQuery2->addToOrderList( std::string( "LHCFILL" ) );
  fillDataQuery2->groupBy( std::string( "LHCFILL" ) );
  //define query output*/
  coral::AttributeList fillDataOutput2;
  fillDataOutput2.extend<float>( std::string( "DELIVEREDLUMI" ) );
  fillDataOutput2.extend<float>( std::string( "RECORDEDLUMI" ) );
  fillDataQuery2->defineOutput( fillDataOutput2 );
  //execute the query
  coral::ICursor& fillDataCursor2 = fillDataQuery2->execute();
  //initialize loop variables
  float delivLumi = 0., recLumi = 0.;
	

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
    fillType = static_cast<LHCInfo::FillTypeId>( fillDataCursor.currentRow()[ std::string( "RUNTIME_TYPE_ID" ) ].data<int>() );
    coral::Attribute const & particleType1Attribute = fillDataCursor.currentRow()[ std::string( "PARTY1" ) ];
    if( particleType1Attribute.isNull() ) {
      particleType1 = LHCInfo::NONE;
    } else {
      particleType1 = static_cast<LHCInfo::ParticleTypeId>( particleType1Attribute.data<int>() );
    }
    coral::Attribute const & particleType2Attribute = fillDataCursor.currentRow()[ std::string( "PARTY2" ) ];
    if( particleType2Attribute.isNull() ) {
      particleType2 = LHCInfo::NONE;
    } else {
      particleType2 = static_cast<LHCInfo::ParticleTypeId>( particleType2Attribute.data<int>() );
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
    
    if( fillDataCursor2.next())
    {
    	coral::Attribute const & delivLumiAttribute = fillDataCursor2.currentRow()[ std::string( "DELIVEREDLUMI" ) ];
		if( delivLumiAttribute.isNull() ){
		  delivLumi = 0.;
		}
		else {
		  delivLumi = delivLumiAttribute.data<float>() / 1000.;
		}
		
		coral::Attribute const & recLumiAttribute = fillDataCursor2.currentRow()[ std::string( "RECORDEDLUMI" ) ];
		if( recLumiAttribute.isNull() ){
		  recLumi = 0.;
		}
		else {
		  recLumi = recLumiAttribute.data<float>() / 1000.;
		}
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
    //if the end time of the fill is 0 (i.e. timestamp null), it is still ongoing: do not store!
    if( beamDumpTime == 0 ) {
      edm::LogWarning( m_name ) << "NO TRANSFER NEEDED: the fill number " << currentFill
				<< " is still ongoing"
				<< "; from " << m_name << "::getNewObjects";
      continue;
    }
    
    //run the third and fourth query against the schema hosting detailed DIP information
    coral::ISchema& beamCondSchema = session.coralSession().schema( m_dipSchema );
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
    bunchConf1Query->limitReturnedRows( LHCInfo::availableBunchSlots ); //maximum number of filled bunches
    bunchConf1Query->defineOutput( bunchConfOutput );
    coral::ICursor& bunchConf1Cursor = bunchConf1Query->execute();
    std::bitset<LHCInfo::bunchSlots+1> bunchConfiguration1( 0ULL );

    while( bunchConf1Cursor.next() ) {
      if( m_debug ) {
	std::ostringstream b1s;
	fillDataCursor.currentRow().toOutputStream( b1s );
	edm::LogInfo( m_name ) << b1s.str() << "\nfrom " << m_name << "::getNewObjects";
      }
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
    bunchConf2Query->limitReturnedRows( LHCInfo::availableBunchSlots ); //maximum number of filled bunches
    bunchConf2Query->defineOutput( bunchConfOutput );
    coral::ICursor& bunchConf2Cursor = bunchConf2Query->execute();
    std::bitset<LHCInfo::bunchSlots+1> bunchConfiguration2( 0ULL );
    
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
      
	//execute query for lumiPerBX
	std::unique_ptr<coral::IQuery> lumiDataQuery(beamCondSchema.newQuery());
	lumiDataQuery->addToTableList( std::string( "CMS_LHC_LUMIPERBUNCH" ), std::string( "LUMIPERBUNCH\", TABLE( LUMIPERBUNCH.LUMI_BUNCHINST ) \"VALUE" ) );
	lumiDataQuery->addToOutputList( std::string( "LUMIPERBUNCH.DIPTIME" ), std::string( "DIPTIME" ) );
	lumiDataQuery->addToOutputList( std::string( "VALUE.COLUMN_VALUE" ), std::string( "LUMI/BUNCH" ) );
	coral::AttributeList lumiDataBindVariables;
	lumiDataBindVariables.extend<coral::TimeStamp>( std::string( "stableBeamStartTimeStamp" ) );
	lumiDataBindVariables[ std::string( "stableBeamStartTimeStamp" ) ].data<coral::TimeStamp>() = stableBeamStartTimeStamp;
	lumiDataBindVariables.extend<coral::TimeStamp>( std::string( "beamDumpTimeStamp" ) );
	lumiDataBindVariables[ std::string( "beamDumpTimeStamp" ) ].data<coral::TimeStamp>() = beamDumpTimeStamp;
	conditionStr = std::string( "DIPTIME BETWEEN :stableBeamStartTimeStamp AND :beamDumpTimeStamp" );
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

	while( lumiDataCursor.next() ) {
	      if( m_debug ) {
		std::ostringstream lpBX;
		lumiDataCursor.currentRow().toOutputStream( lpBX );
		edm::LogInfo( m_name ) << lpBX.str() << "\nfrom " << m_name << "::getNewObjects";
	      }
	      if( lumiDataCursor.currentRow()[ std::string( "VALUE" ) ].data<float>() != 0.00 ) {
		lumiPerBX.push_back(lumiDataCursor.currentRow()[ std::string( "VALUE" ) ].data<float>());
	      }
	}
	  
	//commit the transaction against the DIP "deep" database backend schema
	session.transaction().commit();
	  
	//run the fifth query against the CTPPS schema
	//Initializing the CMS_CTP_CTPPS_COND schema.
	coral::ISchema& CTPPS = session.coralSession().schema("CMS_CTP_CTPPS_COND");
	session.transaction().start( true );
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
	CTPPSDataBindVariables.extend<int>( std::string( "currentFill" ) );
	CTPPSDataBindVariables[ std::string( "currentFill" ) ].data<int>() = currentFill;
	conditionStr = std::string( "FILL_NUMBER = :currentFill" );
	CTPPSDataQuery->setCondition( conditionStr, CTPPSDataBindVariables );
	//ORDER BY clause
	CTPPSDataQuery->addToOrderList( std::string( "DIP_UPDATE_TIME" ) );
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
	std::string lhcState, lhcComment, ctppsStatus;
	unsigned int lumiSection;

	if( CTPPSDataCursor.next() ) {
		if( m_debug ) {
		    std::ostringstream CTPPS;
		    CTPPSDataCursor.currentRow().toOutputStream( CTPPS );
		    edm::LogInfo( m_name ) << CTPPS.str() << "\nfrom " << m_name << "::getNewObjects";
		}
		coral::Attribute const & lhcStateAttribute = CTPPSDataCursor.currentRow()[ std::string( "LHC_STATE" ) ];
		if( lhcStateAttribute.isNull() ) {
			lhcState = "";
		} else {
			lhcState = lhcStateAttribute.data<std::string>();
		}

		coral::Attribute const & lhcCommentAttribute = CTPPSDataCursor.currentRow()[ std::string( "LHC_COMMENT" ) ];
		if( lhcCommentAttribute.isNull() ) {
			lhcComment = "";
		} else {
			lhcComment = lhcCommentAttribute.data<std::string>();
		}

		coral::Attribute const & ctppsStatusAttribute = CTPPSDataCursor.currentRow()[ std::string( "CTPPS_STATUS" ) ];
		if( ctppsStatusAttribute.isNull() ) {
			ctppsStatus = "";
		} else {
			ctppsStatus = ctppsStatusAttribute.data<std::string>();
		}

		coral::Attribute const & lumiSectionAttribute = CTPPSDataCursor.currentRow()[ std::string( "LUMI_SECTION" ) ];
		if( lumiSectionAttribute.isNull() ) {
			lumiSection = 0;
		} else {
			lumiSection = lumiSectionAttribute.data<int>();
		}
	}
	//commit the transaction against the CTPPS schema
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
	//Include this line to insert blank payload between two consecutive fills.
      //m_to_transfer.push_back( std::make_pair( new LHCInfo(), afterPreviousFillEndTime ) );
    } else {
      //the current fill cannot start before the previous one!
      edm::LogError( m_name ) << "WRONG DATA! In the previous fill number " << previousFillNumber
			      << " beams were dumped at timestamp " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( previousFillEndTime ) )
			      << ", which is not before the timestamp " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( stableBeamStartTime ) )
			      << " when current fill number " << currentFill
			      << " entered stable beams. EXITING. from " << m_name << "::getNewObjects";
      return;
    }
    //construct an instance of LHCInfo and set its values
    std::vector<float> dummy(1, 0.); //The ECal vectors will replace the test dummy.
    
    LHCInfo *lhcInfo = new LHCInfo( currentFill ); 
    lhcInfo->setInfo( const_cast<unsigned short const &>( bunches1 )
			 , const_cast<unsigned short const &>( bunches2 )
			 , const_cast<unsigned short const &>( collidingBunches )
			 , const_cast<unsigned short const &>( targetBunches )
			 , const_cast<LHCInfo::FillTypeId const &>( fillType )
			 , const_cast<LHCInfo::ParticleTypeId const &>( particleType1 )
			 , const_cast<LHCInfo::ParticleTypeId const &>( particleType2 )
			 , const_cast<float const &>( crossingAngle )
			 , const_cast<float const &>( betastar )
			 , const_cast<float const &>( intensityBeam1 )
			 , const_cast<float const &>( intensityBeam2 ) 
			 , const_cast<float const &>( energy ) 
			 , const_cast<float const &>( delivLumi )
			 , const_cast<float const &>( recLumi )  
			 , const_cast<cond::Time_t const &>( creationTime )
			 , const_cast<cond::Time_t const &>( stableBeamStartTime )
			 , const_cast<cond::Time_t const &>( beamDumpTime )
			 , const_cast<std::string const &>( injectionScheme )
			 , const_cast<std::vector<float> const &>( lumiPerBX )
			 , const_cast<std::string const &>( lhcState )
			 , const_cast<std::string const &>( lhcComment )
			 , const_cast<std::string const &>( ctppsStatus )
			 , const_cast<unsigned int const &>( lumiSection )
			 , const_cast<std::vector<float> const &>( dummy )
			 , const_cast<std::vector<float> const &>( dummy )
			 , const_cast<std::vector<float> const &>( dummy )
			 , const_cast<std::vector<float> const &>( dummy )
		 	 , const_cast<std::bitset<LHCInfo::bunchSlots+1> const &>( bunchConfiguration1 )
			 , const_cast<std::bitset<LHCInfo::bunchSlots+1> const &>( bunchConfiguration2 )  );
    //store this payload
    m_to_transfer.push_back( std::make_pair( (LHCInfo*) lhcInfo, stableBeamStartTime ) );
    edm::LogInfo( m_name ) << "The new payload to be inserted into tag " << tagInfo().name
			   << " with validity " << stableBeamStartTime 
			   << " ( " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( stableBeamStartTime ) )
			   << " ) has values:\n" << *lhcInfo
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
  //store log information
  m_userTextLog = ss.str();
  edm::LogInfo( m_name ) << "Transferring " << m_to_transfer.size() << " payload(s); from " << m_name << "::getNewObjects";
  }


std::string LHCInfoPopConSourceHandler::id() const { 
  return m_name;
}

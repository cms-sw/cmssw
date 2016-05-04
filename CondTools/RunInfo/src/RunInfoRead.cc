//#include "CondFormats/Common/interface/TimeConversions.h"
//#include "CondFormats/Common/interface/Time.h"
#include "CondTools/RunInfo/interface/RunInfoRead.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
#include <math.h>



namespace {
  std::string dot(".");
  std::string quote("\"");
  std::string bNOTb(" NOT ");
  std::string squoted( const std::string& s ){
    return quote+s+quote;
  }
  //now strings for the tables and columns to be queried
  std::string sParameterTable( "RUNSESSION_PARAMETER" );
  std::string sDateTable( "RUNSESSION_DATE" );
  std::string sStringTable( "RUNSESSION_STRING" );
  std::string sIdParameterColumn( "ID" );
  std::string sRunNumberParameterColumn( "RUNNUMBER" );
  std::string sNameParameterColumn( "NAME" );
  std::string sRunSessionParameterIdDataColumn( "RUNSESSION_PARAMETER_ID" );
  std::string sValueDataColumn( "VALUE" );
  std::string sDCSMagnetTable( "CMSFWMAGNET" );
  std::string sDCSMagnetCurrentColumn( "CURRENT" );
  std::string sDCSMagnetChangeDateColumn( "CHANGE_DATE" );
}

RunInfoRead::RunInfoRead( const std::string& connectionString
			, const edm::ParameterSet& connectionPset ):
   m_connectionString( connectionString )
  ,m_connectionPset( connectionPset ) {}

RunInfoRead::~RunInfoRead() {}

RunInfo 
RunInfoRead::readData( const std::string & runinfo_schema
		     , const std::string & dcsenv_schema
                     , const int r_number ) {
  RunInfo  sum;
  RunInfo temp_sum;
  //for B currents...
  bool Bnotchanged = 0;
  //from TimeConversions.h
  const boost::posix_time::ptime time0 = boost::posix_time::from_time_t(0);
  //if cursor is null setting null values  
  temp_sum.m_run = r_number;
  std::cout << "entering readData" << std::endl;
  cond::persistency::ConnectionPool connection;
  connection.setParameters( m_connectionPset );
  connection.configure();
  edm::LogInfo( "RunInfoReader" ) << "[RunInfoRead::" << __func__ << "]: Initialising read-only session to " << m_connectionString << std::endl;
  boost::shared_ptr<coral::ISessionProxy> session = connection.createCoralSession( m_connectionString, false );
  try{
    session->transaction().start( true );
    std::cout << "starting session " << std::endl;
    coral::ISchema& schema = session->schema( runinfo_schema );
    std::cout << " accessing schema " << runinfo_schema << std::endl;
    //new query to obtain the start_time
    std::unique_ptr<coral::IQuery> query( schema.newQuery() );
    query->addToTableList( sParameterTable );
    query->addToTableList( sDateTable );
    query->addToOutputList( sValueDataColumn );
    coral::AttributeList runTimeDataOutput;
    runTimeDataOutput.extend<coral::TimeStamp>( sValueDataColumn );
    query->defineOutput( runTimeDataOutput );
    std::string runStartWhereClause( sRunNumberParameterColumn + std::string( "=:n_run AND " )
                                     + sNameParameterColumn + std::string( "='CMS.LVL0:START_TIME_T' AND " )
                                     + sIdParameterColumn + std::string( "=" ) + sRunSessionParameterIdDataColumn );
    coral::AttributeList runNumberBindVariableList;
    runNumberBindVariableList.extend<int>( "n_run" );
    runNumberBindVariableList[ "n_run" ].data<int>() = r_number;
    query->setCondition( runStartWhereClause, runNumberBindVariableList );
    coral::ICursor& runStartCursor = query->execute();
    coral::TimeStamp start; //now all times are UTC!
    if( runStartCursor.next() ) {
      //runStartCursor.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = runStartCursor.currentRow();
      start = row[ sValueDataColumn ].data<coral::TimeStamp>();
      /*
      std::cout << "start time extracted == " 
		<< "-->year " << start.year()
		<< "-- month " << start.month()
		<< "-- day " << start.day()
		<< "-- hour " << start.hour()
		<< "-- minute " << start.minute()
		<< "-- second " << start.second()
		<< "-- nanosecond " << start.nanosecond() 
		<< std::endl;
      */
      boost::posix_time::ptime start_ptime = start.time();
      std::cout << "Posix time for run start: "<< start_ptime << std::endl;
      boost::posix_time::time_duration startTimeFromEpoch = start_ptime - time0;
      temp_sum.m_start_time_str = boost::posix_time::to_iso_extended_string(start_ptime);
      temp_sum.m_start_time_ll = startTimeFromEpoch.total_microseconds();
      std::cout << "start time string extracted == " << temp_sum.m_start_time_str << std::endl; 
      std::cout << "microsecond since Epoch (UTC) : " << temp_sum.m_start_time_ll << std::endl;    
    }
    else {
	temp_sum.m_start_time_str = "null";
	temp_sum.m_start_time_ll = -1;
    }
    
    //new query to obtain the stop_time
    query.reset( schema.newQuery() );
    query->addToTableList( sParameterTable );
    query->addToTableList( sDateTable );
    query->addToOutputList( sValueDataColumn );
    query->defineOutput( runTimeDataOutput );
    std::string runStopWhereClause( sRunNumberParameterColumn + std::string( "=:n_run AND " )
                                    + sNameParameterColumn + std::string( "='CMS.LVL0:STOP_TIME_T' AND " )
                                    + sIdParameterColumn + std::string( "=" ) + sRunSessionParameterIdDataColumn );
    query->setCondition( runStopWhereClause, runNumberBindVariableList );
    coral::ICursor& runStopCursor = query->execute();
    coral::TimeStamp stop;
    if( runStopCursor.next() ) {
      //runStopCursor.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = runStopCursor.currentRow();
      stop = row[ sValueDataColumn ].data<coral::TimeStamp>(); 
      /*
      std::cout << "stop time extracted == " 
		<< "-->year " << stop.year()
		<< "-- month " << stop.month()
		<< "-- day " << stop.day()
		<< "-- hour " << stop.hour()
		<< "-- minute " << stop.minute()
		<< "-- second " << stop.second()
		<< "-- nanosecond " << stop.nanosecond() 
		<< std::endl;
      */
      boost::posix_time::ptime stop_ptime = stop.time();
      std::cout << "Posix time for run stop: "<< stop_ptime << std::endl;
      boost::posix_time::time_duration stopTimeFromEpoch = stop_ptime - time0;
      temp_sum.m_stop_time_str = boost::posix_time::to_iso_extended_string(stop_ptime);
      temp_sum.m_stop_time_ll = stopTimeFromEpoch.total_microseconds();
      std::cout << "stop time string extracted == " << temp_sum.m_stop_time_str << std::endl; 
      std::cout << "microsecond since Epoch (UTC) : " << temp_sum.m_stop_time_ll << std::endl;
    }
    else {
      temp_sum.m_stop_time_str = "null";
      temp_sum.m_stop_time_ll = -1;
    }
    
    //new query for obtaining the list of FEDs included in the run
    query.reset( schema.newQuery() );  
    query->addToTableList( sParameterTable );
    query->addToTableList( sStringTable );
    query->addToOutputList( sValueDataColumn );
    query->defineOutputType( sValueDataColumn, "string" );
    std::string fedWhereClause( sRunNumberParameterColumn + std::string( "=:n_run AND " )
                                + sNameParameterColumn + std::string( "='CMS.LVL0:FED_ENABLE_MASK' AND " )
                                + sIdParameterColumn + std::string( "=" ) + sRunSessionParameterIdDataColumn );
    query->setCondition( fedWhereClause, runNumberBindVariableList );
    coral::ICursor& fedCursor = query->execute();
    std::string fed;
    if ( fedCursor.next() ) {
      //fedCursor.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = fedCursor.currentRow();
      fed = row[ sValueDataColumn ].data<std::string>();
    }
    else {
      fed="null";
    }
    //std::cout << "string fed emask == " << fed << std::endl;
    
    std::replace(fed.begin(), fed.end(), '%', ' ');
    std::stringstream stream(fed);
    for(;;) {
      std::string word; 
      if ( !(stream >> word) ){break;}
      std::replace(word.begin(), word.end(), '&', ' ');
      std::stringstream ss(word);
      int fedNumber; 
      int val;
      ss >> fedNumber >> val;
      //std::cout << "fed:: " << fed << "--> val:: " << val << std::endl; 
      //val bit 0 represents the status of the SLINK, but 5 and 7 means the SLINK/TTS is ON but NA or BROKEN (see mail of alex....)
      if( (val & 0001) == 1 && (val != 5) && (val != 7) ) 
	temp_sum.m_fed_in.push_back(fedNumber);
    } 
    std::cout << "feds in run:--> ";
    std::copy(temp_sum.m_fed_in.begin(), temp_sum.m_fed_in.end(), std::ostream_iterator<int>(std::cout, ", "));
    std::cout << std::endl;

    //we connect now to the DCS schema in order to retrieve the magnet current
    coral::ISchema& schema2 = session->schema( dcsenv_schema );
    query.reset( schema2.tableHandle( sDCSMagnetTable ).newQuery() );
    query->addToOutputList( squoted( sDCSMagnetCurrentColumn ), sDCSMagnetCurrentColumn );
    query->addToOutputList( sDCSMagnetChangeDateColumn );
    coral::AttributeList magnetDataOutput;
    magnetDataOutput.extend<float>( sDCSMagnetCurrentColumn );
    magnetDataOutput.extend<coral::TimeStamp>( sDCSMagnetChangeDateColumn );
    query->defineOutput( magnetDataOutput );
    //condition
    coral::AttributeList magnetCurrentBindVariableList;
    float last_current = -1;
    magnetCurrentBindVariableList.extend<coral::TimeStamp>( "runstart_time" );
    magnetCurrentBindVariableList[ "runstart_time" ].data<coral::TimeStamp>() = start;
    std::string magnetCurrentWhereClause;
    if(temp_sum.m_stop_time_str != "null") {
      magnetCurrentBindVariableList.extend<coral::TimeStamp>( "runstop_time" );
      magnetCurrentBindVariableList[ "runstop_time" ].data<coral::TimeStamp>() = stop;
      magnetCurrentWhereClause = std::string( " NOT " ) + squoted(sDCSMagnetCurrentColumn) + std::string( " IS NULL AND " )
                                 + sDCSMagnetChangeDateColumn + std::string( ">:runstart_time AND " ) 
                                 + sDCSMagnetChangeDateColumn + std::string( "<:runstop_time" );
    } else {
      std::cout << "run stop null" << std::endl;
      magnetCurrentWhereClause = std::string( " NOT " ) + squoted(sDCSMagnetCurrentColumn) + std::string( " IS NULL AND " )
                                 + sDCSMagnetChangeDateColumn + std::string( "<:runstart_time" );
    }
    query->setCondition( magnetCurrentWhereClause, magnetCurrentBindVariableList );
    query->addToOrderList( sDCSMagnetChangeDateColumn + std::string( " DESC" ) );
    query->limitReturnedRows( 10000 );
    coral::ICursor& magnetCurrentCursor = query->execute();
    coral::TimeStamp lastCurrentDate;
    std::string last_date;
    std::vector<double> time_curr;
    if ( !magnetCurrentCursor.next() ) {
      // we should deal with stable currents... so the query is returning no value and we should take the last modified current value...
      Bnotchanged = 1;
      std::unique_ptr<coral::IQuery> lastValueQuery( schema2.tableHandle(sDCSMagnetTable).newQuery() );
      lastValueQuery->addToOutputList( squoted(sDCSMagnetCurrentColumn), sDCSMagnetCurrentColumn );
      lastValueQuery->defineOutputType( sDCSMagnetCurrentColumn, "float" );
      coral::AttributeList lastValueBindVariableList;
      lastValueBindVariableList.extend<coral::TimeStamp>( "runstop_time" );
      lastValueBindVariableList[ "runstop_time" ].data<coral::TimeStamp>() = stop;
      std::string lastValueWhereClause( std::string( " NOT " ) + squoted(sDCSMagnetCurrentColumn) + std::string( " IS NULL AND " )
                                        + sDCSMagnetChangeDateColumn + std::string( " <:runstop_time" ) );
      lastValueQuery->setCondition( lastValueWhereClause, lastValueBindVariableList );
      lastValueQuery->addToOrderList( sDCSMagnetChangeDateColumn + std::string( " DESC" ) );
      coral::ICursor& lastValueCursor = lastValueQuery->execute();
      if( lastValueCursor.next() ) {
	//lastValueCursor.currentRow().toOutputStream(std::cout) << std::endl;
	const coral::AttributeList& row = lastValueCursor.currentRow();
	last_current = row[sDCSMagnetCurrentColumn].data<float>();
	std::cout << "previos run(s) current, not changed in this run... " << last_current << std::endl;
      }
      temp_sum.m_avg_current = last_current;
      temp_sum.m_min_current = last_current;
      temp_sum.m_max_current = last_current;
      temp_sum.m_stop_current = last_current;
      temp_sum.m_start_current = last_current; 
    }
    while( magnetCurrentCursor.next() ) {
      //magnetCurrentCursor.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = magnetCurrentCursor.currentRow();
      lastCurrentDate = row[sDCSMagnetChangeDateColumn].data<coral::TimeStamp>();
      temp_sum.m_current.push_back( row[sDCSMagnetCurrentColumn].data<float>() );
      if(temp_sum.m_stop_time_str == "null") break;
      /*
      std::cout << "  last current time extracted == " 
		<< "-->year " << lastCurrentDate.year()
	        << "-- month " << lastCurrentDate.month()
	        << "-- day " << lastCurrentDate.day()
		<< "-- hour " << lastCurrentDate.hour() 
		<< "-- minute " << lastCurrentDate.minute() 
		<< "-- second " << lastCurrentDate.second()
		<< "-- nanosecond " << lastCurrentDate.nanosecond()
		<< std::endl;
      */
      boost::posix_time::ptime lastCurrentDate_ptime = lastCurrentDate.time();
      std::cout << "Posix time for last current time: " << lastCurrentDate_ptime << std::endl;
      boost::posix_time::time_duration lastCurrentDateTimeFromEpoch = lastCurrentDate_ptime - time0;
      last_date = boost::posix_time::to_iso_extended_string(lastCurrentDate_ptime);
      std::cout << "last current time extracted  == " << last_date << std::endl;
      long long last_date_ll = lastCurrentDateTimeFromEpoch.total_microseconds();
      time_curr.push_back(last_date_ll);
    }
    
    size_t csize = temp_sum.m_current.size();
    std::cout << "size of currents  " << csize << std::endl;  
    size_t tsize = time_curr.size(); 
    std::cout << "size of time " << tsize << std::endl;
    if(csize != tsize) { 
      std::cout<< "current and time not filled correctly" << std::endl;
    }
    if(tsize > 1) { 
      temp_sum.m_run_intervall_micros = time_curr.front() - time_curr.back();
    } else { 
      temp_sum.m_run_intervall_micros = 0;
    }
    std::cout << "change current during run interval in microseconds " << temp_sum.m_run_intervall_micros << std::endl;
    
    double wi = 0;
    //std::vector<double> v_wi;
    double sumwixi = 0;
    double sumwi = 0;
    float min = -1;
    float max = -1;
    
    if(csize != 0) {
      min = temp_sum.m_current.front();
      max = temp_sum.m_current.front();
      for(size_t i = 0; i < csize; ++i) {
	std::cout << "--> " << temp_sum.m_current[i] << std::endl;
	if( (tsize > 1) && ( i < csize - 1 ) ) { 
	  wi = (time_curr[i] - time_curr[i+1])  ;
	  temp_sum.m_times_of_currents.push_back(wi);
	  //v_wi.push_back(wi);
	  sumwixi += wi * temp_sum.m_current[i] ;
	  sumwi += wi;
	}  
	min = std::min(min, temp_sum.m_current[i]);
	max = std::max(max, temp_sum.m_current[i]);
      }
      //for (size_t i = 0; i < v_wi.size(); ++i) {
      for (size_t i = 0; i < temp_sum.m_times_of_currents.size(); ++i){
	std::cout << "wi " << temp_sum.m_times_of_currents[i] << std::endl;
      }
      temp_sum.m_start_current = temp_sum.m_current.back(); //temp_sum.m_current[csize - 1];
      std::cout << "--> " << "start cur " << temp_sum.m_start_current << std::endl;
      temp_sum.m_stop_current = temp_sum.m_current.front(); //temp_sum.m_current[0];
      std::cout<< "--> " << "stop cur " << temp_sum.m_stop_current << std::endl;
      if (tsize>1) {
	temp_sum.m_avg_current=sumwixi/sumwi;
      } else { 
	temp_sum.m_avg_current= temp_sum.m_start_current; 
      }
      std::cout<< "--> " << "avg cur  " << temp_sum.m_avg_current << std::endl;
      temp_sum.m_max_current= max;
      std::cout<< "--> " << "max cur  " << temp_sum.m_max_current << std::endl;
      temp_sum.m_min_current= min;
      std::cout<< "--> " << "min cur  " << temp_sum.m_min_current << std::endl;
    } else {
      if (!Bnotchanged) {
	temp_sum.m_avg_current = -1;
	temp_sum.m_min_current = -1;
	temp_sum.m_max_current = -1;
	temp_sum.m_stop_current = -1;
	temp_sum.m_start_current = -1;
      }
    }
    
    std::cout << "temp_sum.m_avg_current " << temp_sum.m_avg_current << std::endl;
    std::cout << "temp_sum.m_min_current " << temp_sum.m_min_current << std::endl;
    std::cout << "temp_sum.m_max_current " << temp_sum.m_max_current << std::endl;
    std::cout << "temp_sum.m_stop_current " << temp_sum.m_stop_current << std::endl;
    std::cout << "temp_sum.m_start_current " << temp_sum.m_start_current << std::endl;
    
    session->transaction().commit();
  }
  catch (const std::exception& e) {
    throw cms::Exception( "RunInfoReader" ) << "[RunInfoRead::" << __func__ << "]: "
                                            << "Unable to create a RunInfo payload. Original Exception:\n"
                                            << e.what() << std::endl;
  }
  
  sum= temp_sum;
  return sum;
}

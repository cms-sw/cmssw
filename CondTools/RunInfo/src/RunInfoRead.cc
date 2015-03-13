//#include "CondFormats/Common/interface/TimeConversions.h"
//#include "CondFormats/Common/interface/Time.h"
#include "CondTools/RunInfo/interface/RunInfoRead.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include <algorithm>
#include <iostream>
#include <iterator>
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


}
RunInfoRead::RunInfoRead(const std::string& connectionString,
			 const std::string& user,
			 const std::string& pass):
  TestBase(),
  m_connectionString( connectionString ),
  m_user( user ),
  m_pass( pass ) {
  m_tableToRead="";
  m_columnToRead="";
}

RunInfoRead::~RunInfoRead() {}

void RunInfoRead::run() {}

RunInfo 
RunInfoRead::readData(const std::string & table, 
		      const std::string &column, const int r_number) {
  m_tableToRead = table; // to be cms_runinfo.runsession_parameter
  m_columnToRead= column;  // to be string_value;
  RunInfo  sum;
  RunInfo temp_sum;
  //RunInfo Sum; 
  //for B currents...
  bool Bnotchanged = 0;
  //from TimeConversions.h
  const boost::posix_time::ptime time0 =
    boost::posix_time::from_time_t(0);
  //if cursor is null setting null values  
  temp_sum.m_run = r_number;
  std::cout << "entering readData" << std::endl;
  coral::ISession* session = this->connect( m_connectionString,
                                            m_user, m_pass );
  try{
    session->transaction().start();
    std::cout << "starting session " << std::endl;
    coral::ISchema& schema = session->schema("CMS_RUNINFO");
    std::cout << " accessing schema " << std::endl;
    std::cout << " trying to handle table ::  " << m_tableToRead << std::endl;
    std::string m_columnToRead_id = "ID";
    long long id_start = 0;
    //new query to obtain the start_time, fist obtaining the id
    coral::IQuery* queryI = schema.tableHandle(m_tableToRead).newQuery();  
    //implementing the query here....... 
    queryI->addToOutputList(m_tableToRead + dot + m_columnToRead_id, m_columnToRead_id);
    //condition 
    coral::AttributeList conditionData;
    conditionData.extend<int>( "n_run" );
    conditionData[0].data<int>() = r_number;
    std::string condition1 = m_tableToRead + ".RUNNUMBER=:n_run AND " +  m_tableToRead +  ".NAME='CMS.LVL0:START_TIME_T'";
    queryI->setCondition(condition1, conditionData);
    coral::ICursor& cursorI = queryI->execute();
    if( cursorI.next() ) {
      //cursorI.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorI.currentRow();
      id_start = row[m_columnToRead_id].data<long long>();
    }
    else {
      id_start = -1;
    }
    //std::cout << "id for start time time extracted == " << id_start << std::endl;
    delete queryI;
    
    //now extracting the start time
    std::string m_tableToRead_date = "RUNSESSION_DATE";
    std::string m_columnToRead_val = "VALUE";
    //new query to obtain the start_time, fist obtaining the id
    coral::IQuery* queryII = schema.tableHandle(m_tableToRead_date).newQuery();  
    //implementing the query here....... 
    queryII->addToOutputList(m_tableToRead_date + dot + m_columnToRead_val, m_columnToRead_val);
    //condition 
    coral::AttributeList conditionData2;
    conditionData2.extend<long long>( "n_id" );
    conditionData2[0].data<long long>() = id_start;
    std::string condition2 = m_tableToRead_date + ".RUNSESSION_PARAMETER_ID=:n_id";
    queryII->setCondition(condition2, conditionData2);
    coral::ICursor& cursorII = queryII->execute();
    coral::TimeStamp start; //now all times are UTC!
    if( cursorII.next() ) {
      //cursorII.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorII.currentRow();
      start = row[m_columnToRead_val].data<coral::TimeStamp>();
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
    delete queryII;
    
    //new query to obtain the stop_time, fist obtaining the id
    coral::IQuery* queryIII = schema.tableHandle(m_tableToRead).newQuery();  
    //implementing the query here....... 
    queryIII->addToOutputList(m_tableToRead + dot + m_columnToRead_id, m_columnToRead_id);
    //condition 
    std::string condition3 = m_tableToRead + ".RUNNUMBER=:n_run AND " + m_tableToRead + ".NAME='CMS.LVL0:STOP_TIME_T'";
    queryIII->setCondition(condition3, conditionData);
    coral::ICursor& cursorIII = queryIII->execute();
    long long id_stop = 0;
    if( cursorIII.next() ) {
      //cursorIII.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorIII.currentRow();
      id_stop = row[m_columnToRead_id].data<long long>();  
    }
    else {
      id_stop = -1;
    }
    //std::cout << "id for stop time time extracted == " << id_stop << std::endl;
    delete queryIII;
    
    //now exctracting the stop time
    coral::IQuery* queryIV = schema.tableHandle(m_tableToRead_date).newQuery(); 
    //implementing the query here....... 
    queryIV->addToOutputList(m_tableToRead_date + dot + m_columnToRead_val, m_columnToRead_val);
    //condition
    coral::AttributeList conditionData4;
    conditionData4.extend<long long>( "n_id" );
    conditionData4[0].data<long long>() = id_stop;
    std::string condition4 = m_tableToRead_date + ".RUNSESSION_PARAMETER_ID=:n_id";
    queryIV->setCondition(condition4, conditionData4);
    coral::ICursor& cursorIV = queryIV->execute();
    coral::TimeStamp stop;
    if( cursorIV.next() ) {
      //cursorIV.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorIV.currentRow();
      stop = row[m_columnToRead_val].data<coral::TimeStamp>(); 
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
    delete queryIV;
    
    std::string m_tableToRead_fed = "RUNSESSION_STRING";
    coral::IQuery* queryV = schema.newQuery();  
    queryV->addToTableList(m_tableToRead);
    queryV->addToTableList(m_tableToRead_fed);
    queryV->addToOutputList(m_tableToRead_fed + dot + m_columnToRead_val, m_columnToRead_val);
    //queryV->addToOutputList(m_tableToRead + dot + m_columnToRead, m_columnToRead);
    //condition
    std::string condition5 = m_tableToRead + ".RUNNUMBER=:n_run AND " + m_tableToRead + ".NAME='CMS.LVL0:FED_ENABLE_MASK' AND RUNSESSION_PARAMETER.ID = RUNSESSION_STRING.RUNSESSION_PARAMETER_ID";
    //std::string condition5 = m_tableToRead + ".runnumber=:n_run AND " + m_tableToRead + ".name='CMS.LVL0:FED_ENABLE_MASK'";
    queryV->setCondition(condition5, conditionData);
    coral::ICursor& cursorV = queryV->execute();
    std::string fed;
    if ( cursorV.next() ) {
      //cursorV.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorV.currentRow();
      fed = row[m_columnToRead_val].data<std::string>();
    }
    else {
      fed="null";
    }
    //std::cout << "string fed emask == " << fed << std::endl;
    delete queryV;
    
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
    /*
    for (size_t i =0; i<temp_sum.m_fed_in.size() ; ++i){
      std::cout << "fed in run:--> " << temp_sum.m_fed_in[i] << std::endl; 
    } 
    */
    
    coral::ISchema& schema2 = session->schema("CMS_DCS_ENV_PVSS_COND");
    std::string m_tableToRead_cur= "CMSFWMAGNET";
    std::string m_columnToRead_cur= "CURRENT";
    std::string m_columnToRead_date= "CHANGE_DATE";
    coral::IQuery* queryVI = schema2.tableHandle(m_tableToRead_cur).newQuery();
    queryVI->addToOutputList(m_tableToRead_cur +  dot + squoted(m_columnToRead_cur), m_columnToRead_cur);
    queryVI->addToOutputList(m_tableToRead_cur +  dot + m_columnToRead_date, m_columnToRead_date);
    //condition 
    coral::AttributeList conditionData6;
    float last_current = -1;
    if(temp_sum.m_stop_time_str != "null") { 
      conditionData6.extend<coral::TimeStamp>( "runstart_time" );
      conditionData6.extend<coral::TimeStamp>( "runstop_time" );
      conditionData6["runstart_time"].data<coral::TimeStamp>() = start; //start_time ;
      conditionData6["runstop_time"].data<coral::TimeStamp>() = stop; //stop_time ;
      std::string conditionVI = " NOT " + m_tableToRead_cur + dot + squoted(m_columnToRead_cur) + " IS NULL AND " 
	+ m_tableToRead_cur +  dot + m_columnToRead_date + ">:runstart_time AND " 
	+ m_tableToRead_cur +  dot + m_columnToRead_date + "<:runstop_time"  /*" ORDER BY " + m_columnToRead_date + " DESC"*/;
      queryVI->setCondition(conditionVI, conditionData6);
      queryVI->addToOrderList(m_tableToRead_cur +  dot + m_columnToRead_date + " DESC");
    } else {
      std::cout << "run stop null" << std::endl;
      conditionData6.extend<coral::TimeStamp>( "runstart_time" );
      conditionData6["runstart_time"].data<coral::TimeStamp>() = start; //start_time ;
      std::string conditionVI = " NOT " + m_tableToRead_cur + dot + squoted(m_columnToRead_cur) + " IS NULL AND " 
	+ m_tableToRead_cur +  dot + m_columnToRead_date + "<:runstart_time" /*" ORDER BY " + m_columnToRead_date + " DESC"*/;
      queryVI->setCondition(conditionVI, conditionData6);
      queryVI->addToOrderList(m_tableToRead_cur +  dot + m_columnToRead_date + " DESC");
    }
    queryVI->limitReturnedRows(10000);
    coral::ICursor& cursorVI = queryVI->execute();
    coral::TimeStamp lastCurrentDate;
    std::string last_date;
    std::vector<double> time_curr;
    if ( !cursorVI.next() ) {
      // we should deal with stable currents... so the query is returning no value and we should take the last modified current value...
      Bnotchanged = 1;
      coral::AttributeList conditionData6bis;
      conditionData6bis.extend<coral::TimeStamp>( "runstop_time" );
      conditionData6bis["runstop_time"].data<coral::TimeStamp>() = stop; //stop_time ;
      std::string conditionVIbis = " NOT " + m_tableToRead_cur + dot + squoted(m_columnToRead_cur) + " IS NULL AND " 
	+ m_tableToRead_cur +  dot + m_columnToRead_date + " <:runstop_time" /*" ORDER BY " + m_columnToRead_date + " DESC"*/;
      coral::IQuery* queryVIbis = schema2.tableHandle(m_tableToRead_cur).newQuery();
      queryVIbis->addToOutputList(m_tableToRead_cur + dot +  squoted(m_columnToRead_cur), m_columnToRead_cur);
      queryVIbis->setCondition(conditionVIbis, conditionData6bis);
      queryVIbis->addToOrderList(m_tableToRead_cur +  dot + m_columnToRead_date + " DESC");
      coral::ICursor& cursorVIbis= queryVIbis->execute();
      if( cursorVIbis.next() ) {
	//cursorVIbis.currentRow().toOutputStream(std::cout) << std::endl;
	const coral::AttributeList& row = cursorVIbis.currentRow();
	last_current = row[m_columnToRead_cur].data<float>();
	std::cout << "previos run(s) current, not changed in this run... " << last_current << std::endl;
      } 
      temp_sum.m_avg_current = last_current;
      temp_sum.m_min_current = last_current;
      temp_sum.m_max_current = last_current;
      temp_sum.m_stop_current = last_current;
      temp_sum.m_start_current = last_current; 
    }
    while( cursorVI.next() ) {
      //cursorVI.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursorVI.currentRow();
      lastCurrentDate = row[m_columnToRead_date].data<coral::TimeStamp>();
      temp_sum.m_current.push_back( row[m_columnToRead_cur].data<float>() );
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
    delete queryVI;
    
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
    std::cout << "Exception: " << e.what() << std::endl;
  }
  delete session;
  
  sum= temp_sum;
  return sum;
}

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
//#include "SealBase/TimeInfo.h"

#include "CondCore/DBCommon/interface/Time.h"

#include "CoralBase/TimeStamp.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time.hpp"
  
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>





RunInfoRead::RunInfoRead(
			     
              const std::string& connectionString,
              const std::string& user,
              const std::string& pass):
  TestBase(),
  
  m_connectionString( connectionString ),
  m_user( user ),
  m_pass( pass )
{

  m_tableToRead="";
  m_columnToRead="";

}


RunInfoRead::~RunInfoRead()
{}



void
RunInfoRead::run()
{
  
}





RunInfo::RunInfo 
RunInfoRead::readData(const std::string & table, const std::string &column, const int r_number)
{
  m_tableToRead = table; // to be  cms_runinfo.runsession_parameter
  m_columnToRead= column;  // to be string_value;
  
  
  
  RunInfo::RunInfo  sum;
  RunInfo::RunInfo temp_sum;
  RunInfo Sum; 



  // if cursor is null  setting null values  
  temp_sum.m_run = r_number;

  std::cout<< "entering readData" << std::endl;
  coral::ISession* session = this->connect( m_connectionString,
                                            m_user, m_pass );
  session->transaction().start( );
  std::cout<< "starting session " << std::endl;
  coral::ISchema& schema = session->nominalSchema();
  std::cout<< " accessing schema " << std::endl;
  std::cout<< " trying to handle table ::  " << m_tableToRead << std::endl;
  // coral::IQuery* queryI = schema.newQuery();
  

  
  //  condition 
 
 coral::AttributeList conditionData;
  conditionData.extend<int>( "n_run" );
 conditionData[0].data<int>() = r_number;

 // for B currents...
 bool Bnotchanged=0;
  

     
   std::string m_columnToRead_id = "ID";
   long long id_start=0;
   // new query to obtain the start_time, fist obtaining the id
   coral::IQuery* queryI = schema.tableHandle( m_tableToRead).newQuery();  
   //queryIII->addToTableList( m_tableToRead );
   // implemating the query here....... 
   queryI->addToOutputList( m_tableToRead + "." +  m_columnToRead_id, m_columnToRead_id    );
   //  condition 
   std::string condition1 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:START_TIME_T'";
   queryI->setCondition( condition1, conditionData );
   coral::ICursor& cursorI = queryI->execute();
   
 if ( cursorI.next()!=0 ) {
   const coral::AttributeList& row = cursorI.currentRow();
   id_start= row[m_columnToRead_id].data<long long>();
   std::cout<< " id for  start time time extracted == " <<id_start   << std::endl;
 }
 else{
   id_start=-1;
   std::cout<< " id for  start time time extracted == " <<id_start   << std::endl;
 }
 
 delete queryI;
 
 // now exctracting the start time
 std::string m_tableToRead_date= "RUNSESSION_DATE";
 std::string m_columnToRead_val= "VALUE";
 // new query to obtain the start_time, fist obtaining the id
 coral::IQuery* queryII = schema.tableHandle( m_tableToRead_date).newQuery();  
 //queryII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryII->addToOutputList( m_tableToRead_date + "." +  m_columnToRead_val, m_columnToRead_val    );
  //  condition 
 coral::AttributeList conditionData2;
 conditionData2.extend<long long>( "n_id" );
 conditionData2[0].data<long long>() = id_start;
 std::string condition2 = m_tableToRead_date + ".runsession_parameter_id=:n_id";
 queryII->setCondition( condition2, conditionData2 );
 coral::ICursor& cursorII = queryII->execute();
 coral::TimeStamp start;
 coral::TimeStamp start_time;
 if ( cursorII.next()!=0 ) {
   const coral::AttributeList& row = cursorII.currentRow();
   start =  row[m_columnToRead_val].data<coral::TimeStamp>();    
   int  year= start.year();
   int  month= start.month();
   int  day= start.day();
   int  hour= start.hour();
   int  minute= start.minute();
   int  second = start.second();
   long nanosecond =  start.nanosecond();
   //const std::string toString= start.toString() ;    
   /// The number of nanoseconds from epoch 01/01/1970 UTC, normally should fit into 64bit signed integer, depends on the BOOST installation
   //const signed long long int  total_nanoseconds=start.total_nanoseconds() ;

 //adjust to UTC (shift 27/10) fix in march 09........   
   int  adj_hour=0;
if ( month <= 10 && day<=27) {
      adj_hour= hour-2;}else{
      adj_hour= hour-1;
     }

 
  start_time= coral::TimeStamp(year, month, day, adj_hour, minute, second , nanosecond); 

   std::cout<< "  start time time extracted == " << "-->year " << year
	    << "-- month " << month
	    << "-- day " << day
	    << "-- hour " << hour 
	    << "-- adj_hour " << adj_hour 
	    << "-- minute " << minute 
	    << "-- second " << second
	    << "-- nanosecond " << nanosecond<<std::endl;
   boost::gregorian::date dt(year,month,day);
   // td in microsecond
   boost::posix_time::time_duration td(adj_hour,minute,second,nanosecond/1000);  
   

   boost::posix_time::ptime pt( dt, td); 

  

   //boost::gregorian::date(year,month,day),
   //boost::posix_time::hours(hour)+boost::posix_time::minutes(minute)+ 
   //boost::posix_time::seconds(second)+ 
   //nanosec(nanosecond));
   // boost::posix_time::ptime pt(start);
   std::cout<<"ptime == "<< pt <<std::endl;          
   
   temp_sum.m_start_time_str = boost::posix_time::to_iso_extended_string(pt);
   std::cout<<"start time string  extracted  == "<<temp_sum.m_start_time_str   <<std::endl;   
   boost::posix_time::ptime time_at_epoch( boost::gregorian::date( 1970, 1, 1 ) ) ;
   // Subtract time_at_epoch from current time to get the required value.
   boost::posix_time::time_duration time_diff = ( pt - time_at_epoch ) ;
   temp_sum.m_start_time_ll = time_diff.total_microseconds();
   std::cout << "microsecond since Epoch (UTC) : " <<temp_sum.m_start_time_ll  <<std::endl;    
 }
 else 
   {
     temp_sum.m_start_time_str = "null";
     temp_sum.m_start_time_ll = -1;
   }
 delete queryII;
 
 
 // new query to obtain the stop_time, fist obtaining the id
 coral::IQuery* queryIII = schema.tableHandle( m_tableToRead).newQuery();  
 //queryII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIII->addToOutputList( m_tableToRead + "." +  m_columnToRead_id, m_columnToRead_id    );
 //  condition 
 std::string condition3 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:STOP_TIME_T'";
 
 queryIII->setCondition( condition3, conditionData );
 
 coral::ICursor& cursorIII = queryIII->execute();
 
 
 long long id_stop=0;
 if ( cursorIII.next()!=0 ) {
   const coral::AttributeList& row = cursorIII.currentRow();
        
   id_stop= row[m_columnToRead_id].data<long long>();
   std::cout<< " id for  stop time time extracted == " <<id_stop   << std::endl;
 }
 else{
  id_stop=-1;
 }
 delete queryIII;
 
 // now exctracting the start time
 // new query to obtain the start_time, fist obtaining the id
 coral::IQuery* queryIV = schema.tableHandle( m_tableToRead_date).newQuery(); 
 //queryIII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIV->addToOutputList( m_tableToRead_date + "." +  m_columnToRead_val, m_columnToRead_val    );
 //  condition 
 coral::AttributeList conditionData4;
 conditionData4.extend<long long>( "n_id" );
 conditionData4[0].data<long long>() = id_stop;
 std::string condition4 = m_tableToRead_date + ".runsession_parameter_id=:n_id";
 coral::TimeStamp stop;
coral::TimeStamp stop_time;


 queryIV->setCondition( condition4, conditionData4 );
 coral::ICursor& cursorIV = queryIV->execute();
 if ( cursorIV.next()!=0 ) {
   const coral::AttributeList& row = cursorIV.currentRow();
   stop =  row[m_columnToRead_val].data<coral::TimeStamp>(); 
   int  year= stop.year();
   int  month= stop.month();
   int  day= stop.day();
   int  hour= stop.hour();
   int  minute= stop.minute();
   int  second = stop.second();
   long nanosecond =  stop.nanosecond();
  
//adjust to UTC (shift 27/10 02:00) fix in march 09........
  int  adj_hour=0;
 if ( month <= 10 && day<=27 ) {
     adj_hour= hour-2;}else{
    adj_hour= hour-1;
     }
   stop_time=coral::TimeStamp(year, month, day, adj_hour, minute, second , nanosecond);
 


    std::cout<< "  stop time time extracted == " << "-->year " << year
	    << "-- month " << month
	    << "-- day " << day
	    << "-- hour " << hour 
	    << "-- adj_hour " << adj_hour 
	    << "-- minute " << minute 
	    << "-- second " << second
	    << "-- nanosecond " << nanosecond<<std::endl;
   boost::gregorian::date dt(year,month,day);
   boost::posix_time::time_duration td(adj_hour,minute,second,nanosecond/1000);  
   boost::posix_time::ptime pt( dt, td); 
   std::cout<<"ptime == "<< pt <<std::endl;          
   temp_sum.m_stop_time_str = boost::posix_time::to_iso_extended_string(pt);
   std::cout<<"stop time string  extracted  == "<<temp_sum.m_stop_time_str   <<std::endl;   
   boost::posix_time::ptime time_at_epoch( boost::gregorian::date( 1970, 1, 1 ) ) ;
   // Subtract time_at_epoch from current time to get the required value.
   boost::posix_time::time_duration time_diff = ( pt - time_at_epoch ) ;
   temp_sum.m_stop_time_ll = time_diff.total_microseconds();
   std::cout << "microsecond since Epoch (UTC) : " <<temp_sum.m_stop_time_ll  <<std::endl;   

 
 }
 else{
   temp_sum.m_stop_time_str="null";
   temp_sum.m_stop_time_ll=-1;
 }
 delete queryIV;

 
   coral::IQuery* queryV = schema.tableHandle( m_tableToRead).newQuery();  
   
   queryV->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead    );
   //  cond
  

   std::string condition5 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:FED_ENABLE_MASK'";
   queryV->setCondition( condition5, conditionData );
   coral::ICursor& cursorV = queryV->execute();
   std::string fed;
 if ( cursorV.next()!=0 ) {
   const coral::AttributeList& row = cursorV.currentRow();
   fed= row[m_columnToRead].data<std::string>();
   // std::cout<< " string fed emask  == " << fed   << std::endl;
 }
 else{
   fed="null";
   //  std::cout<< " string fed emask  == " << fed   << std::endl;
 }
 
 delete queryV;
 
 std::replace(fed.begin(), fed.end(), '%', ' ');
 std::stringstream stream(fed);
 for(;;) {
   std::string word; 
   if (!(stream>> word)){break;}
   std::replace(word.begin(), word.end(), '&', ' ');
   std::stringstream ss(word);
   int fed; int val;
   ss>>fed>>val;
   //  std::cout <<" fed:: "<<fed<<"--> val:: "<<val<<std::endl; 
   // val bit 0 represents the status of the SLINK, but5 and 7 means the SLINK/TTS is ON but NA or BROKEN (see mail of alex....)
  
   if ( (val & 0001) ==1 && (val!=5) && (val!=7) )  temp_sum.m_fed_in.push_back(fed);
  } 


 for (size_t i =0; i<temp_sum.m_fed_in.size() ;i++){
   //  std::cout<< "fed in run" << temp_sum.m_fed_in[i] << std::endl; 
}  




 coral::ISchema& schema2 = session->schema("CMS_DCS_ENV_PVSS_COND");

 std::string m_tableToRead_cur= "CMSFWMAGNET";
 std::string m_columnToRead_cur= "CURRENT";
 std::string m_columnToRead_date= "CHANGE_DATE";

 coral::IQuery* queryVI = schema2.tableHandle( m_tableToRead_cur).newQuery();

queryVI->addToOutputList(   m_tableToRead_cur +  "." +  m_columnToRead_cur , m_columnToRead_cur  );
 queryVI->addToOutputList( m_tableToRead_cur +  "." +  m_columnToRead_date , m_columnToRead_date  );

  //  condition 
coral::AttributeList conditionData6;

 float last_current=-1;
 
 if (temp_sum.m_stop_time_str!="null") { 
   conditionData6.extend<coral::TimeStamp>( "runstart_time" );
   conditionData6.extend<coral::TimeStamp>( "runstop_time" );
   conditionData6["runstart_time"].data<coral::TimeStamp>() = start_time ;
   conditionData6["runstop_time"].data<coral::TimeStamp>() = stop_time ;
   std::string conditionVI = " NOT " + m_tableToRead_cur + "." + m_columnToRead_cur + " IS NULL  AND "+ m_columnToRead_date+ ">:runstart_time AND "+ m_columnToRead_date + "<:runstop_time "  "ORDER BY " + m_columnToRead_date +  " DESC";
queryVI->setCondition( conditionVI , conditionData6 );
} else {
 std::cout<< "run stop null " << std::endl;
   conditionData6.extend<coral::TimeStamp>( "runstart_time" );
   conditionData6["runstart_time"].data<coral::TimeStamp>() = start_time ;
  std::string conditionVI = " NOT " + m_tableToRead_cur + "." + m_columnToRead_cur + " IS NULL AND "+ m_columnToRead_date+ "<:runstart_time " "ORDER BY " + m_columnToRead_date +  " DESC";
   queryVI->setCondition( conditionVI , conditionData6 );


 }

  queryVI->limitReturnedRows( 10000 );
  coral::ICursor& cursorVI = queryVI->execute();

  std::string last_date;

  std::vector<double> time_curr;
 
  if (cursorVI.next()==0  ){

  
    // we should deal with stable currents..... so the quesry is returning no value and we should take the last modified currnt value.....
    Bnotchanged=1;
   
    coral::AttributeList conditionData6bis;
    conditionData6bis.extend<coral::TimeStamp>( "runstop_time" );
    
    conditionData6bis["runstop_time"].data<coral::TimeStamp>() = stop_time ;
    
    std::string conditionVIbis = " NOT " + m_tableToRead_cur + "." + m_columnToRead_cur + " IS NULL AND "+ m_columnToRead_date+ " <:runstop_time" " ORDER BY " + m_columnToRead_date +  " DESC";
    
    coral::IQuery* queryVIbis = schema2.tableHandle( m_tableToRead_cur).newQuery();

    queryVIbis->addToOutputList(   m_tableToRead_cur +  "." +  m_columnToRead_cur , m_columnToRead_cur  );
 queryVIbis->setCondition( conditionVIbis , conditionData6bis);

 coral::ICursor& cursorVIbis= queryVIbis->execute();
 if ( cursorVIbis.next()!=0  ) {
   
   const coral::AttributeList& row = cursorVIbis.currentRow();
   last_current = row[m_columnToRead_cur].data<float>();
   std::cout<< "previos run(s) current, not changed in this run... " << last_current << std::endl;
 }   
  temp_sum.m_avg_current=last_current;
  temp_sum.m_min_current=last_current;
  temp_sum.m_max_current=last_current;
  temp_sum.m_stop_current=last_current;
  temp_sum.m_start_current=last_current; 
  }
  

  while ( cursorVI.next()!=0  ) {
    const coral::AttributeList& row = cursorVI.currentRow();
    coral::TimeStamp ld = row[m_columnToRead_date].data<coral::TimeStamp>();
    
    temp_sum.m_current.push_back( row[m_columnToRead_cur].data<float>());
    if (temp_sum.m_stop_time_str=="null") break;

    
    
     int  year= ld.year();
   int  month= ld.month();
   int  day= ld.day();
   int  hour= ld.hour();
   int  minute= ld.minute();
   int  second = ld.second();
   long nanosecond =  ld.nanosecond();
   //const std::string toString= ld.toString() ;    
   /// The number of nanoseconds from epoch 01/01/1970 UTC, normally should fit into 64bit signed integer, depends on the BOOST installation
   //const signed long long int  total_nanoseconds=ld.total_nanoseconds() ;
   
   
   //  std::cout<< "  start time time extracted == " << "-->year " << year
   //    << "-- month " << month
   //	    << "-- day " << day
   //	    << "-- hour " << hour 
   //	    << "-- minute " << minute 
   //	    << "-- second " << second
   //	    << "-- nanosecond " << nanosecond<<std::endl;
   
   boost::gregorian::date dt(year,month,day);
   // td in microsecond
   boost::posix_time::time_duration td(hour,minute,second,nanosecond/1000);  
   boost::posix_time::ptime pt( dt, td); 
     
   //std::cout<<"ptime == "<< pt <<std::endl;          
   
   last_date = boost::posix_time::to_iso_extended_string(pt);
    std::cout<<"last current time  extracted  == "<<last_date   <<std::endl;   
   boost::posix_time::ptime time_at_epoch( boost::gregorian::date( 1970, 1, 1 ) ) ;
   // Subtract time_at_epoch from current time to get the required value.
   boost::posix_time::time_duration time_diff = ( pt - time_at_epoch ) ;
   long long last_date_ll = time_diff.total_microseconds();
   time_curr.push_back(last_date_ll);  
  //std::cout << "microsecond since Epoch (UTC) : " << last_date_ll  <<std::endl;    


  }
  
  
   delete queryVI;
   
   size_t size= temp_sum.m_current.size();
   std::cout<< "size of currents  "  <<size<< std::endl;  
   size_t tsize= time_curr.size(); 
   std::cout<< "size of time "  << tsize<< std::endl;
   if (size !=tsize ) { std::cout<< "current and time not filled correctely " << std::endl;}
   
   if ( tsize > 1 ) { temp_sum.m_run_intervall_micros = time_curr[0] - time_curr[tsize-1];} else {  temp_sum.m_run_intervall_micros=0;}
   
   std::cout<< "change currunt during run intervall in microseconds " << temp_sum.m_run_intervall_micros << std::endl;
   
   double wi=0;
   std::vector<double> v_wi;
   double sumwixi=0;
   double sumwi=0;
   float min=-1;
   float max=-1;
   
  
  if (size!=0 ){
      min=temp_sum.m_current[0];
      max=temp_sum.m_current[0];
      for(size_t i=0; i< temp_sum.m_current.size(); i++){
	std::cout<< "--> " << temp_sum.m_current[i] << std::endl;
	if (tsize >1 && ( i < temp_sum.m_current.size()-1 )) { 
	  wi =  (time_curr[i] - time_curr[i+1])  ;
	  //  temp_sum.m_times_of_currents.push_back(wi);
	  v_wi.push_back(wi);
	  sumwixi+= wi * temp_sum.m_current[i] ;
	  sumwi += wi;
	}  
	min= std::min(min, temp_sum.m_current[i]);
	max= std::max(max, temp_sum.m_current[i]);
      }
      
      for (size_t i =0; i<v_wi.size(); i++){
	//for (size_t i =0; i<temp_sum.m_times_of_currents.size(); i++){
	//	std::cout<<"wi "<<temp_sum.m_times_of_currents[i]<<std::endl;
      }
      temp_sum.m_start_current=(temp_sum.m_current[0]) ;
      std::cout<< "--> " << "start cur  "<< temp_sum.m_start_current << std::endl;
      
      temp_sum.m_stop_current=temp_sum.m_current[size-1];
      std::cout<< "--> " << "stop cur  "<< temp_sum.m_stop_current << std::endl;
      if (tsize>1 ) {temp_sum.m_avg_current=sumwixi/sumwi ;}
      else { temp_sum.m_avg_current= temp_sum.m_start_current; }
      std::cout<< "--> " << "avg cur  "<< temp_sum.m_avg_current << std::endl;
      temp_sum.m_max_current= max;
      std::cout<< "--> " << "max cur  "<< temp_sum.m_max_current << std::endl;
      temp_sum.m_min_current= min;
      std::cout<< "--> " << "min cur  "<< temp_sum.m_min_current << std::endl;
   }else{
     if (Bnotchanged==0){
     temp_sum.m_avg_current=-1;
     temp_sum.m_min_current=-1;
     temp_sum.m_max_current=-1;
     temp_sum.m_stop_current=-1;
     temp_sum.m_start_current=-1;
   }
   }
  
  std::cout<< " temp_sum.m_avg_current" << temp_sum.m_avg_current<< std::endl;
  std::cout<< " temp_sum.m_min_current" << temp_sum.m_min_current<< std::endl;
  std::cout<< " temp_sum.m_max_current" << temp_sum.m_max_current<< std::endl;
  std::cout<< " temp_sum.m_stop_current" << temp_sum.m_stop_current<< std::endl;
  std::cout<< " temp_sum.m_start_current" << temp_sum.m_start_current<< std::endl;



 session->transaction().commit();
 delete session;
 
 
 sum= temp_sum;
 return sum;
}




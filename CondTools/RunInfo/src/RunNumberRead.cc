#include "CondTools/RunInfo/interface/RunNumberRead.h"



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
#include "SealBase/TimeInfo.h"

#include "CondCore/DBCommon/interface/Time.h"

#include "CoralBase/TimeStamp.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time.hpp"
  
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>





RunNumberRead::RunNumberRead(
			     
              const std::string& connectionString,
              const std::string& user,
              const std::string& pass):
  TestBase(),
  
  m_connectionString( connectionString ),
  m_user( user ),
  m_pass( pass )
{
  m_tableToDrop="";  
  m_tableToRead="";
  m_columnToRead="";

}


RunNumberRead::~RunNumberRead()
{}



void
RunNumberRead::run()
{
  
}

void RunNumberRead::dropTable(const std::string & table){
  m_tableToDrop = table;
  coral::ISession* session = this->connect( m_connectionString,m_user, m_pass );
   session->transaction().start( );
    std::cout << "connected succesfully to omds" <<std::endl;
    coral::ISchema& schema = session->nominalSchema(); 
    schema.dropIfExistsTable( m_tableToDrop );
    
}




std::vector<RunNumber::Item> 
RunNumberRead::readData(const std::string & table, const std::string &column, const int r_number)
{
  m_tableToRead = table; // to be  cms_runinfo.runsession_parameter
  m_columnToRead= column;  // to be string_value;
  
  /* query to execute:
     1) to extract the seq_name 
     select  string_value from cms_runinfo.runsession_parameter where cms_runinfo.runsession_parameter.runnumber=r_number AND   cms_runinfo.runsession_parameter.name='CMS.LVL0:SEQ_NAME'
     
2) to extract the number 
select string_value from cms_runinfo.runsession_parameter where cms_runinfo.runsession_parameter.runnumber=runnumber AND   cms_runinfo.runsession_parameter.name='CMS.LVL0:SEQ_NUMBER'

3) to extract the start_time 

select id from cms_runinfo.runsession_parameter where runnumber=45903 and cms_runinfo.runsession_parameter.name='CMS.LVL0:START_TIME_T';

and then 
select value from  runsession_date where runsession_parameter_id=1647783


4) the same for stop_time
 
select string_value from cms_runinfo.runsession_parameter where cms_runinfo.runsession_parameter.runnumber=runnumber AND   cms_runinfo.runsession_parameter.name='CMS.LVL0:STOP_TIME_T'

5) to extract the lumisections number for the run
select MAX(lsnumber)   FROM cms_runinfo.hlt_supervisor_lumisections where cms_runinfo.hlt_supervisor_lumisections.runnr=runnumber
  */
  

  


  std::cout<< "entering readData" << std::endl;
  coral::ISession* session = this->connect( m_connectionString,
                                            m_user, m_pass );
  session->transaction().start( );
  std::cout<< "starting session " << std::endl;
  coral::ISchema& schema = session->nominalSchema();
  std::cout<< " accessing schema " << std::endl;
  std::cout<< " trying to handle table ::  " << m_tableToRead << std::endl;
  // coral::IQuery* queryI = schema.newQuery();
  coral::IQuery* queryI = schema.tableHandle( m_tableToRead).newQuery();

 //  queryI->addToTableList( m_tableToRead );
  std::cout<< "table handling " << std::endl;
  // implemating the query here....... 
  queryI->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string  condition = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:SEQ_NUMBER'";
  coral::AttributeList conditionData;
  conditionData.extend<int>( "n_run" );
  queryI->setCondition( condition, conditionData );
  conditionData[0].data<int>() = r_number;
  coral::ICursor& cursorI = queryI->execute();
 
  /*  class RunNumber {
public:
struct Item {
Item(){}
~Item(){}
int m_run;
long long  m_id_start;
long long  m_id_stop;
std::string m_number;
std::string m_name;
signed long long m_start_time_sll;
std::string m_start_time_str;
signed long long m_stop_time_sll;
std::string m_stop_time_str;
int  m_lumisections;
.....

  */


  std::vector<RunNumber::Item> rnarray;
  RunNumber::Item Itemp;
  

  // if cursor is null  setting null values  
  Itemp.m_run = r_number;
  
  if ( cursorI.next()!=0 ) {
    const coral::AttributeList& row = cursorI.currentRow();
    std::cout<< " entering the query == " << std::endl;
    Itemp.m_number = row[m_columnToRead].data<std::string>();
    std::cout<< " number extracted == " << Itemp.m_number  << std::endl;
  }
  else {
    Itemp.m_number="null";
    std::cout<< " null number extracted == " << Itemp.m_number  << std::endl;
  }
  
  delete queryI;
  
  // new query to obtain the string value
  coral::IQuery* queryII = schema.tableHandle( m_tableToRead).newQuery();  
  //tableHandle( m_tableToRead).newQuery();
  //queryII->addToTableList( m_tableToRead );
  // implemating the query here....... 
  queryII->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition2 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:SEQ_NAME'";
  queryII->setCondition( condition2, conditionData );
  coral::ICursor& cursorII = queryII->execute();
  
  if ( cursorII.next()!=0  ) {
    const coral::AttributeList& row = cursorII.currentRow();
    Itemp.m_name = row[m_columnToRead].data<std::string>();
    std::cout<< " name extracted == " <<Itemp.m_name   << std::endl;
  }
  else{
    Itemp.m_name = "null";
    std::cout<< " name extracted == " <<Itemp.m_name   << std::endl;
  }
   delete queryII;
   
   
   std::string m_columnToRead_id = "ID";
   // new query to obtain the start_time, fist obtaining the id
   coral::IQuery* queryIII = schema.tableHandle( m_tableToRead).newQuery();  
   //queryIII->addToTableList( m_tableToRead );
   // implemating the query here....... 
   queryIII->addToOutputList( m_tableToRead + "." +  m_columnToRead_id, m_columnToRead_id    );
   //  condition 
   std::string condition3 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:START_TIME_T'";
   queryIII->setCondition( condition3, conditionData );
   coral::ICursor& cursorIII = queryIII->execute();
   
 if ( cursorIII.next()!=0 ) {
   const coral::AttributeList& row = cursorIII.currentRow();
   Itemp.m_id_start= row[m_columnToRead_id].data<long long>();
   std::cout<< " id for  start time time extracted == " <<Itemp.m_id_start   << std::endl;
 }
 else{
   Itemp.m_id_start=-1;
   std::cout<< " id for  start time time extracted == " <<Itemp.m_id_start   << std::endl;
 }
 
 delete queryIII;
 
 // now exctracting the start time
 std::string m_tableToRead_date= "RUNSESSION_DATE";
 std::string m_columnToRead_val= "VALUE";
 // new query to obtain the start_time, fist obtaining the id
 coral::IQuery* queryIIIbis = schema.tableHandle( m_tableToRead_date).newQuery();  
 //queryIII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIIIbis->addToOutputList( m_tableToRead_date + "." +  m_columnToRead_val, m_columnToRead_val    );
  //  condition 
 coral::AttributeList conditionData3bis;
 conditionData3bis.extend<long long>( "n_id" );
 conditionData3bis[0].data<long long>() = Itemp.m_id_start;
 std::string condition3bis = m_tableToRead_date + ".runsession_parameter_id=:n_id";
 queryIIIbis->setCondition( condition3bis, conditionData3bis );
 coral::ICursor& cursorIIIbis = queryIIIbis->execute();
 
 if ( cursorIIIbis.next()!=0 ) {
   const coral::AttributeList& row = cursorIIIbis.currentRow();
   coral::TimeStamp ts =  row[m_columnToRead_val].data<coral::TimeStamp>();    
   int  year= ts.year();
   int  month= ts.month();
   int  day= ts.day();
   int  hour= ts.hour();
   int  minute= ts.minute();
   int  second = ts.second();
   long nanosecond =  ts.nanosecond();
   //const std::string toString= ts.toString() ;    
   /// The number of nanoseconds from epoch 01/01/1970 UTC, normally should fit into 64bit signed integer, depends on the BOOST installation
   //const signed long long int  total_nanoseconds=ts.total_nanoseconds() ;
   
   
   std::cout<< "  start time time extracted == " << "-->year " << year
	    << "-- month " << month
	    << "-- day " << day
	    << "-- hour " << hour 
	    << "-- minute " << minute 
	    << "-- second " << second
	    << "-- nanosecond " << nanosecond<<std::endl;
   boost::gregorian::date dt(year,month,day);
   // td in microsecond
   boost::posix_time::time_duration td(hour,minute,second,nanosecond/1000);  
   
   boost::posix_time::ptime pt( dt, td); 
   //boost::gregorian::date(year,month,day),
   //boost::posix_time::hours(hour)+boost::posix_time::minutes(minute)+ 
   //boost::posix_time::seconds(second)+ 
   //nanosec(nanosecond));
   // boost::posix_time::ptime pt(ts);
   std::cout<<"ptime == "<< pt <<std::endl;          
   
   Itemp.m_start_time_str = boost::posix_time::to_iso_extended_string(pt);
   std::cout<<"start time string  extracted  == "<<Itemp.m_start_time_str   <<std::endl;   
   boost::posix_time::ptime time_at_epoch( boost::gregorian::date( 1970, 1, 1 ) ) ;
   // Subtract time_at_epoch from current time to get the required value.
   boost::posix_time::time_duration time_diff = ( pt - time_at_epoch ) ;
   Itemp.m_start_time_sll = time_diff.total_microseconds();
   std::cout << "microsecond since Epoch (UTC) : " <<Itemp.m_start_time_sll  <<std::endl;    
 }
 else 
   {
     Itemp.m_start_time_str = "null";
     Itemp.m_start_time_sll = -1;
   }
 delete queryIIIbis;
 
 
 // new query to obtain the stop_time, fist obtaining the id
 coral::IQuery* queryIV = schema.tableHandle( m_tableToRead).newQuery();  
 //queryIII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIV->addToOutputList( m_tableToRead + "." +  m_columnToRead_id, m_columnToRead_id    );
 //  condition 
 std::string condition4 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:STOP_TIME_T'";
 
 queryIV->setCondition( condition4, conditionData );
 
 coral::ICursor& cursorIV = queryIV->execute();
 
 
 
 if ( cursorIV.next()!=0 ) {
   const coral::AttributeList& row = cursorIV.currentRow();
        
   Itemp.m_id_stop= row[m_columnToRead_id].data<long long>();
   std::cout<< " id for  stop time time extracted == " <<Itemp.m_id_stop   << std::endl;
 }
 else{
   Itemp.m_id_stop=-1;
 }
 delete queryIV;
 
 // now exctracting the start time
 // new query to obtain the start_time, fist obtaining the id
 coral::IQuery* queryIVbis = schema.tableHandle( m_tableToRead_date).newQuery(); 
 //queryIII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIVbis->addToOutputList( m_tableToRead_date + "." +  m_columnToRead_val, m_columnToRead_val    );
 //  condition 
 coral::AttributeList conditionData4bis;
 conditionData4bis.extend<long long>( "n_id" );
 conditionData4bis[0].data<long long>() = Itemp.m_id_stop;
 std::string condition4bis = m_tableToRead_date + ".runsession_parameter_id=:n_id";
 queryIVbis->setCondition( condition4bis, conditionData4bis );
 coral::ICursor& cursorIVbis = queryIVbis->execute();
 if ( cursorIVbis.next()!=0 ) {
   const coral::AttributeList& row = cursorIVbis.currentRow();
   coral::TimeStamp ts =  row[m_columnToRead_val].data<coral::TimeStamp>(); 
   int  year= ts.year();
   int  month= ts.month();
   int  day= ts.day();
   int  hour= ts.hour();
   int  minute= ts.minute();
   int  second = ts.second();
   long nanosecond =  ts.nanosecond();
   std::cout<< "  stop time time extracted == " << "-->year " << year
	    << "-- month " << month
	    << "-- day " << day
	    << "-- hour " << hour 
	    << "-- minute " << minute 
	    << "-- second " << second
	    << "-- nanosecond " << nanosecond<<std::endl;
   boost::gregorian::date dt(year,month,day);
   boost::posix_time::time_duration td(hour,minute,second,nanosecond/1000);  
   boost::posix_time::ptime pt( dt, td); 
   std::cout<<"ptime == "<< pt <<std::endl;          
   Itemp.m_stop_time_str = boost::posix_time::to_iso_extended_string(pt);
   std::cout<<"stop time string  extracted  == "<<Itemp.m_stop_time_str   <<std::endl;   
   boost::posix_time::ptime time_at_epoch( boost::gregorian::date( 1970, 1, 1 ) ) ;
   // Subtract time_at_epoch from current time to get the required value.
   boost::posix_time::time_duration time_diff = ( pt - time_at_epoch ) ;
   Itemp.m_stop_time_sll = time_diff.total_microseconds();
   std::cout << "microsecond since Epoch (UTC) : " <<Itemp.m_stop_time_sll  <<std::endl;    
 }
 else{
   Itemp.m_stop_time_str="null";
   Itemp.m_stop_time_sll=-1;
 }
 delete queryIVbis;
 
 // new query to obtain the lumisections number
 const std::string m_tableToRead_ls="HLT_SUPERVISOR_LUMISECTIONS";
 const std::string m_columnToRead_ls = "LSNUMBER";
 
 coral::IQuery* queryV = schema.tableHandle( m_tableToRead_ls ).newQuery(); 
 queryV->addToOutputList( "MAX(" + m_tableToRead_ls + "." +  m_columnToRead_ls +")", m_columnToRead_ls  );
 std::string condition5 = m_tableToRead_ls + ".runnr=:n_run " ;
 //coral::AttributeList conditionData5;
 //conditionData5.extend<double>( "n_run" );
 queryV->setCondition( condition5, conditionData );
 // queryV->setCondition(std::string("max_lsnumber"),coral::AttributeList() );
 // queryV->defineOutputType( m_columnToRead_ls, "double" );
 coral::ICursor& cursorV = queryV->execute();
 if ( cursorV.next()!=0  ) {
   const coral::AttributeList& row = cursorV.currentRow();
   double lumisections =  row[m_columnToRead_ls].data<double>();
   Itemp.m_lumisections = static_cast<int>(lumisections);
   std::cout<<" lumisections number extracted == "<<Itemp.m_lumisections  << std::endl;  
 }
 else{
   Itemp.m_lumisections=-1; 
 }
 std::cout<<" leaving the query  "<< std::endl;  
 delete queryV;
 
 session->transaction().commit();
 delete session;
 
 
 rnarray.push_back(Itemp);
 return rnarray;
}




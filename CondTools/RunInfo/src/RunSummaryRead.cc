#include "CondTools/RunInfo/interface/RunSummaryRead.h"



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

#include "CondCore/CondDB/interface/Time.h"

#include "CoralBase/TimeStamp.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time.hpp"
  
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>





RunSummaryRead::RunSummaryRead(
			     
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


RunSummaryRead::~RunSummaryRead()
{}



void
RunSummaryRead::run()
{
  
}





RunSummary 
RunSummaryRead::readData(const std::string & table, const std::string &column, const int r_number)
{
  m_tableToRead = table; // to be  cms_runinfo.runsession_parameter
  m_columnToRead= column;  // to be string_value;
  
  /* query to execute:
     1) to extract the seq_name 
     select  string_value from cms_runinfo.runsession_parameter where cms_runinfo.runsession_parameter.runnumber=r_number AND   cms_runinfo.runsession_parameter.name='CMS.LVL0:SEQ_NAME'
     

3) to extract the start_time 

select id from cms_runinfo.runsession_parameter where runnumber=45903 and cms_runinfo.runsession_parameter.name='CMS.LVL0:START_TIME_T';

and then 
select value from  runsession_date where runsession_parameter_id=1647783


4) the same for stop_time
 
select string_value from cms_runinfo.runsession_parameter where cms_runinfo.runsession_parameter.runnumber=runnumber AND   cms_runinfo.runsession_parameter.name='CMS.LVL0:STOP_TIME_T'

5) to extract the lumisections number for the run
select MAX(lsnumber)   FROM cms_runinfo.hlt_supervisor_lumisections where cms_runinfo.hlt_supervisor_lumisections.runnr=runnumber

6) for extract subdt_joined:
select  string_value from cms_runinfo.runsession_parameter   where cms_runinfo.runsession_parameter.runnumber=51770 AND cms_runinfo.runsession_parameter.name LIKE 'CMS.LVL0%' RPC, ECAL,....
  */  
  
  RunSummary  sum;
  RunSummary temp_sum;
  RunSummary Sum; 



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
  coral::IQuery* queryI = schema.tableHandle( m_tableToRead).newQuery();

 queryI->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
 
 coral::AttributeList conditionData;
  conditionData.extend<int>( "n_run" );
 conditionData[0].data<int>() = r_number;


  //  condition 
  std::string condition1 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:SEQ_NAME'";
  queryI->setCondition( condition1, conditionData );
  coral::ICursor& cursorI = queryI->execute();
  
  if ( cursorI.next()!=0  ) {
    const coral::AttributeList& row = cursorI.currentRow();
     temp_sum.m_name = row[m_columnToRead].data<std::string>();
    std::cout<< " name extracted == " << temp_sum.m_name   << std::endl;
  }
  else{
     temp_sum.m_name = "null";
    std::cout<< " name extracted == " << temp_sum.m_name   << std::endl;
  }
   delete queryI;
   
   
   std::string m_columnToRead_id = "ID";
   long long id_start=0;
   // new query to obtain the start_time, fist obtaining the id
   coral::IQuery* queryII = schema.tableHandle( m_tableToRead).newQuery();  
   //queryIII->addToTableList( m_tableToRead );
   // implemating the query here....... 
   queryII->addToOutputList( m_tableToRead + "." +  m_columnToRead_id, m_columnToRead_id    );
   //  condition 
   std::string condition2 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:START_TIME_T'";
   queryII->setCondition( condition2, conditionData );
   coral::ICursor& cursorII = queryII->execute();
   
 if ( cursorII.next()!=0 ) {
   const coral::AttributeList& row = cursorII.currentRow();
   id_start= row[m_columnToRead_id].data<long long>();
   std::cout<< " id for  start time time extracted == " <<id_start   << std::endl;
 }
 else{
   id_start=-1;
   std::cout<< " id for  start time time extracted == " <<id_start   << std::endl;
 }
 
 delete queryII;
 
 // now exctracting the start time
 std::string m_tableToRead_date= "RUNSESSION_DATE";
 std::string m_columnToRead_val= "VALUE";
 // new query to obtain the start_time, fist obtaining the id
 coral::IQuery* queryIII = schema.tableHandle( m_tableToRead_date).newQuery();  
 //queryIII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIII->addToOutputList( m_tableToRead_date + "." +  m_columnToRead_val, m_columnToRead_val    );
  //  condition 
 coral::AttributeList conditionData3;
 conditionData3.extend<long long>( "n_id" );
 conditionData3[0].data<long long>() = id_start;
 std::string condition3 = m_tableToRead_date + ".runsession_parameter_id=:n_id";
 queryIII->setCondition( condition3, conditionData3 );
 coral::ICursor& cursorIII = queryIII->execute();
 
 if ( cursorIII.next()!=0 ) {
   const coral::AttributeList& row = cursorIII.currentRow();
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
   // td in microsecond, fix to utc.....
   boost::posix_time::time_duration td(hour-1,minute,second,nanosecond/1000);  
   
   boost::posix_time::ptime pt( dt, td); 
   //boost::gregorian::date(year,month,day),
   //boost::posix_time::hours(hour)+boost::posix_time::minutes(minute)+ 
   //boost::posix_time::seconds(second)+ 
   //nanosec(nanosecond));
   // boost::posix_time::ptime pt(ts);
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
 delete queryIII;
 
 
 // new query to obtain the stop_time, fist obtaining the id
 coral::IQuery* queryIV = schema.tableHandle( m_tableToRead).newQuery();  
 //queryIII->addToTableList( m_tableToRead );
 // implemating the query here....... 
 queryIV->addToOutputList( m_tableToRead + "." +  m_columnToRead_id, m_columnToRead_id    );
 //  condition 
 std::string condition4 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:STOP_TIME_T'";
 
 queryIV->setCondition( condition4, conditionData );
 
 coral::ICursor& cursorIV = queryIV->execute();
 
 
 long long id_stop=0;
 if ( cursorIV.next()!=0 ) {
   const coral::AttributeList& row = cursorIV.currentRow();
        
   id_stop= row[m_columnToRead_id].data<long long>();
   std::cout<< " id for  stop time time extracted == " <<id_stop   << std::endl;
 }
 else{
  id_stop=-1;
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
 conditionData4bis[0].data<long long>() = id_stop;
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
   // fix to utc....
   boost::posix_time::time_duration td(hour-1,minute,second,nanosecond/1000);  
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
 delete queryIVbis;
 
 // new query to obtain the lumisections number
 const std::string m_tableToRead_ls="HLT_SUPERVISOR_LUMISECTIONS_V2";
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
   temp_sum.m_lumisections = static_cast<int>(lumisections);
   std::cout<<" lumisections number extracted == "<<temp_sum.m_lumisections  << std::endl;  
 }
 else{
   temp_sum.m_lumisections=-1; 
 }
 std::cout<<" leaving the query  "<< std::endl;  
 delete queryV;





 
 // new queries to obtain the subdetector joining or not

coral::IQuery* queryVIPIXEL = schema.tableHandle( m_tableToRead).newQuery();  
  queryVIPIXEL->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6PIXEL = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:PIXEL'";
  queryVIPIXEL->setCondition( condition6PIXEL, conditionData );
  coral::ICursor& cursorVIPIXEL = queryVIPIXEL->execute();
  
  if ( cursorVIPIXEL.next()!=0  ) {
    const coral::AttributeList& row = cursorVIPIXEL.currentRow();
    //      temp_sum.m_subdt_joining.push_back("PIXEL:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.PIXEL) ;
      }
   else{
    //   temp_sum.m_subdt_joining.push_back("PIXEL:null");
    
   }
   delete queryVIPIXEL;

coral::IQuery* queryVITRACKER = schema.tableHandle( m_tableToRead).newQuery();  
  queryVITRACKER->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6TRACKER = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:TRACKER'";
  queryVITRACKER->setCondition( condition6TRACKER, conditionData );
  coral::ICursor& cursorVITRACKER = queryVITRACKER->execute();
  
  if ( cursorVITRACKER.next()!=0  ) {
    const coral::AttributeList& row = cursorVITRACKER.currentRow();
    
    //  temp_sum.m_subdt_joining.push_back("TRACKER:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.TRACKER) ;
  }
  else{
    // temp_sum.m_subdt_joining.push_back("TRACKER:null");
    
  }
   delete queryVITRACKER;

coral::IQuery* queryVIECAL = schema.tableHandle( m_tableToRead).newQuery();  
  queryVIECAL->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6ECAL = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:ECAL'";
  queryVIECAL->setCondition( condition6ECAL, conditionData );
  coral::ICursor& cursorVIECAL = queryVIECAL->execute();
  
  if ( cursorVIECAL.next()!=0  ) {
    const coral::AttributeList& row = cursorVIECAL.currentRow();
    
    // temp_sum.m_subdt_joining.push_back("ECAL:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.ECAL) ;
  }
  else{
    // temp_sum.m_subdt_joining.push_back("ECAL:null");
    
  }
   delete queryVIECAL;

coral::IQuery* queryVIHCAL = schema.tableHandle( m_tableToRead).newQuery();  
  queryVIHCAL->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6HCAL = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:HCAL'";
  queryVIHCAL->setCondition( condition6HCAL, conditionData );
  coral::ICursor& cursorVIHCAL = queryVIHCAL->execute();
  
  if ( cursorVIHCAL.next()!=0  ) {
    const coral::AttributeList& row = cursorVIHCAL.currentRow();
    
    //temp_sum.m_subdt_joining.push_back("HCAL:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.HCAL) ;
  }
  else{
    // temp_sum.m_subdt_joining.push_back("HCAL:null");
    
  }
   delete queryVIHCAL;


coral::IQuery* queryVIDT = schema.tableHandle( m_tableToRead).newQuery();  
  queryVIDT->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6DT = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:DT'";
  queryVIDT->setCondition( condition6DT, conditionData );
  coral::ICursor& cursorVIDT = queryVIDT->execute();
  
  if ( cursorVIDT.next()!=0  ) {
    const coral::AttributeList& row = cursorVIDT.currentRow();
    
    //  temp_sum.m_subdt_joining.push_back("DT:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.DT) ;
  }
  else{
    //   temp_sum.m_subdt_joining.push_back("DT:null");
    
  }
   delete queryVIDT;

coral::IQuery* queryVICSC = schema.tableHandle( m_tableToRead).newQuery();  
  queryVICSC->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6CSC = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:CSC'";
  queryVICSC->setCondition( condition6CSC, conditionData );
  coral::ICursor& cursorVICSC = queryVICSC->execute();
  
  if ( cursorVICSC.next()!=0  ) {
    const coral::AttributeList& row = cursorVICSC.currentRow();
    
    //  temp_sum.m_subdt_joining.push_back("CSC:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.CSC) ;
  }
  else{
    //   temp_sum.m_subdt_joining.push_back("CSC:null");
    
  }
   delete queryVICSC;

coral::IQuery* queryVIRPC = schema.tableHandle( m_tableToRead).newQuery();  
  queryVIRPC->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition6RPC = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:RPC'";
  queryVIRPC->setCondition( condition6RPC, conditionData );
  coral::ICursor& cursorVIRPC = queryVIRPC->execute();
  
  if ( cursorVIRPC.next()!=0  ) {
    const coral::AttributeList& row = cursorVIRPC.currentRow();
    
    //  temp_sum.m_subdt_joining.push_back("RPC:" + row[m_columnToRead].data<std::string>());
    if (row[m_columnToRead].data<std::string>()=="In") temp_sum.m_subdt_in.push_back(Sum.RPC) ;
  }
  else{
    //   temp_sum.m_subdt_joining.push_back("RPC:null");
    
  }
   delete queryVIRPC;

 
   //  for (size_t pos=0; pos<  temp_sum.m_subdt_joining.size(); ++pos){ 
   //   std::cout <<" value for subdetector joined extracted " <<temp_sum.m_subdt_joining[pos] << std::endl;
   //   }

   // new query to catch hlt key
   coral::IQuery* queryVII = schema.tableHandle( m_tableToRead).newQuery();

 queryVII->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
 
 coral::AttributeList conditionData7;
  conditionData7.extend<int>( "n_run" );
 conditionData7[0].data<int>() = r_number;


  //  condition 
  std::string condition7 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:HLT_KEY_DESCRIPTION'";
  queryVII->setCondition( condition7, conditionData7 );
  coral::ICursor& cursorVII = queryVII->execute();
  
  if ( cursorVII.next()!=0  ) {
    const coral::AttributeList& row = cursorVII.currentRow();
     temp_sum.m_hltkey = row[m_columnToRead].data<std::string>();
    std::cout<< " hlt key extracted == " << temp_sum.m_hltkey   << std::endl;
  }
  else{
    temp_sum.m_hltkey = "null";
    std::cout<< " hlt key extracted == " << temp_sum.m_hltkey   << std::endl;
  }
   delete queryVII;
  
   // new query to catch event number
   coral::IQuery* queryVIII = schema.tableHandle( m_tableToRead).newQuery();

 queryVIII->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
 
 coral::AttributeList conditionData8;
  conditionData8.extend<int>( "n_run" );
 conditionData8[0].data<int>() = r_number;


  //  condition 
  std::string condition8 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.TRG:EVNR' ORDER BY TIME DESC";
  queryVIII->setCondition( condition8, conditionData8 );
  coral::ICursor& cursorVIII = queryVIII->execute();
  
  if ( cursorVIII.next()!=0  ) {
    const coral::AttributeList& row = cursorVIII.currentRow();

    temp_sum.m_nevents=atoll(row[m_columnToRead].data<std::string>().c_str());
     
   
    std::cout<< " number of events extracted == " << temp_sum.m_nevents   << std::endl;
  }
  else{
    temp_sum.m_nevents = -1;
    std::cout<< " number of events extracted == " << temp_sum.m_nevents   << std::endl;
  }
   delete queryVIII;
    
 // new query to catch event number
   coral::IQuery* queryIX = schema.tableHandle( m_tableToRead).newQuery();

 queryIX->addToOutputList( "AVG(" + m_tableToRead + "." +  m_columnToRead + ")", m_columnToRead );
  //  condition 
 

 coral::AttributeList conditionData9;
  conditionData9.extend<int>( "n_run" );
 conditionData9[0].data<int>() = r_number;


  //  condition 
  std::string condition9 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.TRG:Rate' ORDER BY TIME";

  queryIX->setCondition( condition9, conditionData9 );
   coral::ICursor& cursorIX = queryIX->execute();
  
  if ( cursorIX.next()!=0  ) {
    const coral::AttributeList& row = cursorIX.currentRow();
     
    temp_sum.m_rate=(float)row[m_columnToRead].data<double>();
     
   
    std::cout<< " rate extracted == " << temp_sum.m_rate   << std::endl;
  }
  else{
    temp_sum.m_rate = -1;
    std::cout<< " rate  extracted == " << temp_sum.m_rate   << std::endl;
  }
   delete queryIX;




 session->transaction().commit();
 delete session;
 
 
 sum= temp_sum;
 return sum;
}




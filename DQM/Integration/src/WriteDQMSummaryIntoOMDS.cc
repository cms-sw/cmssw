#include "DQM/Integration/interface/WriteDQMSummaryIntoOMDS.h"

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
#include<fstream>
#include<vector>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>

WriteDQMSummaryIntoOMDS::WriteDQMSummaryIntoOMDS(
           
              const std::string& connectionString,
              const std::string& user,
              const std::string& pass):
  TestBase(),
  m_connectionString( connectionString ),
  m_user( user ),
  m_pass( pass )
{
  m_tableToDrop="";  
  m_viewToDrop="";  
  m_tableToAppend="";
  m_file="";

}


WriteDQMSummaryIntoOMDS::~WriteDQMSummaryIntoOMDS()
{}



void
WriteDQMSummaryIntoOMDS::run()
{
}

void WriteDQMSummaryIntoOMDS::dropTable(const std::string & table){
  m_tableToDrop = table;
coral::ISession* session = this->connect( m_connectionString,m_user, m_pass );
   session->transaction().start( );
    std::cout << "connected succesfully to omds" <<std::endl;
   coral::ISchema& schema = session->nominalSchema(); 
  schema.dropIfExistsTable(m_tableToDrop);
 
 
}

void WriteDQMSummaryIntoOMDS::dropView(const std::string & view){
  m_viewToDrop = view;
  coral::ISession* session = this->connect( m_connectionString,m_user, m_pass );
  session->transaction().start( );
  std::cout << "connected succesfully to omds" <<std::endl;
  coral::ISchema& schema = session->nominalSchema(); 
  schema.dropIfExistsView(m_viewToDrop);
}


std::vector<Item>
WriteDQMSummaryIntoOMDS::readData(const std::string & file)
{
  
  m_file = file;
  std::ifstream indata;
  indata.open(m_file.c_str(),std::ios::in);
  if(!indata) {
    std::cerr <<"Error: no such file!"<< std::endl;
    exit(1);
  }
  Item ItemI_tmp;
 char line[100];
 char first_three_char[3];
  // string of the file relevant  
 std::string spaces="===";
 std::string Run="Run";
 // string needed for parsing 
 char subsys[50], comp[50], tmp[3];
 int l=0;
   //reading the file and filling the struct
  while(!indata.eof()) {
    indata.getline(line,100);
    l++;
    for(int k=0; k<3;k++){
      first_three_char[k]= line[k];
    }
    // skipp fist line and when ===
    if ( l==1  || first_three_char ==spaces) continue;
    if (first_three_char==Run){
      // getting run number and subdetector name 
      sscanf( line, "%s%d%s", tmp , &ItemI_tmp.m_run , subsys);
      ItemI_tmp.m_subsystem = subsys;
    } else {
      // getting the string/component between the < .... > and the status
      sscanf( line, "<%[^>]>%f",comp,&ItemI_tmp.m_status );
      ItemI_tmp.m_reportcontent=comp;
         m_itemvec.push_back(ItemI_tmp);
       }
  }
  indata.close();
  // to avoid inserted last row twice..... 
  m_itemvec.pop_back();
  return  m_itemvec;
}
 
 

void
WriteDQMSummaryIntoOMDS::writeData(const std::string & tableToRead)
  {
    
 for (size_t k= 0; k< m_itemvec.size(); k++){
    Item Itemtemp= m_itemvec[k];
    // printing to stdout the value to be transferred 
  std::cout<< "read from file, and now transferring  these value   : " << std::endl;
  std::cout<< "item run  : " << Itemtemp.m_run<<std::endl;
  std::cout<< "item subsystem : " << Itemtemp.m_subsystem << std::endl;
  std::cout<< "item reportcontent  : " << Itemtemp.m_reportcontent <<std::endl;
  std::cout<< "item status  : " << Itemtemp.m_status<<std::endl;

  }
 
  
  std::cout<< "starting to write the data in omds  : " << std::endl;
  m_tableToAppend = tableToRead; 

  std::cout << "connecting to omds" <<std::endl;
 coral::ISession* session = this->connect( m_connectionString,m_user, m_pass );
 session->transaction().start( );
 std::cout << "connected succesfully to omds" <<std::endl;
 coral::ISchema& schema = session->nominalSchema(); 
 
 std::cout << "new table: " <<  m_tableToAppend <<std::endl;
 coral::ITable& table = schema.tableHandle(m_tableToAppend); 
 coral::AttributeList rowBuffer;
 table.dataEditor().rowBuffer( rowBuffer );
// appending to the existing table
 for (size_t k= 0; k< m_itemvec.size(); k++ )
   {
     Item Itemtemp= m_itemvec[k];

     rowBuffer["RUN"].data<long long int>() =Itemtemp.m_run ;
     rowBuffer["SUBSYSTEM"].data<std::string>() = Itemtemp.m_subsystem;
     rowBuffer["REPORTCONTENT"].data<std::string>() = Itemtemp.m_reportcontent;
     rowBuffer["STATUS"].data<double>() =  Itemtemp.m_status;
     table.dataEditor().insertRow( rowBuffer ); 
   }
 
 session->transaction().commit();
delete session;
 
}




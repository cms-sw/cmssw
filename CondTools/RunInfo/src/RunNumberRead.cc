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

  */

  std::cout<< "entering readData" << std::endl;
  coral::ISession* session = this->connect( m_connectionString,
                                            m_user, m_pass );
  session->transaction().start( );
  std::cout<< "starting session " << std::endl;
  coral::ISchema& schema = session->nominalSchema();
  std::cout<< " accessing schema " << std::endl;
  std::cout<< " trying to handle table ::  " << m_tableToRead << std::endl;
  coral::IQuery* queryI = schema.newQuery();
 //coral::IQuery* queryI = schema.tableHandle( m_tableToRead).newQuery();



  queryI->addToTableList( m_tableToRead );
  std::cout<< "table handling " << std::endl;
// implemating the query here....... 
  queryI->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:SEQ_NUMBER'";

  coral::AttributeList conditionData;
  conditionData.extend<int>( "n_run" );
  
 queryI->setCondition( condition, conditionData );
  
  conditionData[0].data<int>() = r_number;
   
 coral::ICursor& cursorI = queryI->execute();
 
 
   std::vector<RunNumber::Item> rnarray;
 
   RunNumber::Item Itemp;
   
   while ( cursorI.next() ) {
    const coral::AttributeList& row = cursorI.currentRow();
    
    Itemp.m_number = row[m_columnToRead].data<std::string>();
     
    Itemp.m_name = "";
    Itemp.m_index = r_number ;
    std::cout<< " number extracted == " << Itemp.m_number  << std::endl;
   
   rnarray.push_back(Itemp);
    }
   delete queryI;
   // new query to obtain the string value
  coral::IQuery* queryII = schema.newQuery();
  //tableHandle( m_tableToRead).newQuery();

queryII->addToTableList( m_tableToRead );
// implemating the query here....... 
  queryII->addToOutputList( m_tableToRead + "." +  m_columnToRead, m_columnToRead  );
  //  condition 
  std::string condition2 = m_tableToRead + ".runnumber=:n_run AND " +  m_tableToRead +  ".name='CMS.LVL0:SEQ_NAME'";

 queryII->setCondition( condition2, conditionData );
  
 coral::ICursor& cursorII = queryII->execute();

 
     size_t i=0;
   while ( cursorII.next() || i< rnarray.size() ) {
    const coral::AttributeList& row = cursorII.currentRow();
        
    rnarray[i].m_name = row[m_columnToRead].data<std::string>();
   std::cout<< " name extracted == " <<rnarray[i].m_name   << std::endl;
   i++;
       
      }
   delete queryII;



   
  session->transaction().commit();
  delete session;
  return rnarray;
}




#include "CondTools/DQM/interface/DQMSummaryReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RelationalAccess/ISession.h"
//#include "RelationalAccess/ISessionProxy.h"
//#include "RelationalAccess/ITypeConverter.h"
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
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

DQMSummaryReader::DQMSummaryReader(const std::string& connectionString,
				   const std::string& user,
				   const std::string& pass):
  TestBase(), /*ReadBase(),*/
  m_connectionString( connectionString ),
  m_user( user ),
  m_pass( pass ) {
  m_tableToRead="";
}

DQMSummaryReader::~DQMSummaryReader() {}

void DQMSummaryReader::run() {}

DQMSummary DQMSummaryReader::readData(const std::string & table, /*const std::string & column,*/ const long long r_number) {
  m_tableToRead = table; // to be  CMS_DQM_SUMMARY.summarycontent
  //m_columnToRead = column;  // to be run, lumisec
  DQMSummary dqmSummary;
  dqmSummary.m_run = r_number;
  std::cout<< "Entering readData" << std::endl;
  coral::ISession* session = this->connect(m_connectionString,
                                           m_user, m_pass);
  //coral::ISessionProxy* session = this->connect(m_connectionString,
  //                                              m_user, m_pass);
  try{
    //session->typeConverter().setCppTypeForSqlType(coral::AttributeSpecification::typeNameForId(typeid(std::string)), "VARCHAR2(20)");
    //session->typeConverter().setCppTypeForSqlType(coral::AttributeSpecification::typeNameForId(typeid(std::string)), "VARCHAR2(40)");
    session->transaction().start();
    std::cout<< "Starting session on the connection " << m_connectionString << std::endl;
    coral::ISchema& schema = session->nominalSchema();
    std::cout<< "--- accessing schema for user " << m_user << std::endl;
    std::cout<< "------ trying to handle table " << m_tableToRead << std::endl;
    //defining count query
    coral::IQuery* query = schema.tableHandle(m_tableToRead).newQuery();
    query->addToOutputList("count(*)", "count");
    //condition for the WHERE clause in the count query
    std::string condition = "run=:n_run";
    coral::AttributeList conditionData;
    conditionData.extend<long long>( "n_run" );
    //conditionData[0].setValue<long long>(r_number); 
    conditionData[0].data<long long>() = r_number;
    query->setCondition(condition, conditionData);
    //performing count query
    coral::ICursor& cursor = query->execute();
    DQMSummary::DQMSummary::RunItem runItem;
    DQMSummary::DQMSummary::RunItem::LumiItem lumiItem;
    double nRows = 0;
    if( cursor.next() ) {
      //cursor.currentRow().toOutputStream(std::cout) << std::endl;
      const coral::AttributeList& row = cursor.currentRow();
      nRows = row["count"].data<double>();
      /*const coral::Attribute& count = cursor.currentRow()["count"];
	if(count.specification().type() == typeid(double))
      nRows = count.data<double>();
      else
      nRows = count.data<float>();*/
      std::cout << "Rows for count query " << nRows << std::endl;
      if( nRows != 0 ) {
	std::cout << "Starting to build DQMSummary" << std::endl;
	//defining first query
	coral::IQuery* queryI = schema.tableHandle(m_tableToRead).newQuery();
	queryI->setDistinct();
	queryI->addToOutputList("lumisec");
	//condition for the WHERE clause in the first query
	std::string conditionI = "run=:n_run";
	coral::AttributeList conditionDataI;
	conditionDataI.extend<long long>( "n_run" );
	//conditionDataI[0].setValue<long long>(r_number); 
	conditionDataI[0].data<long long>() = r_number;
	queryI->setCondition(conditionI, conditionDataI);
	//performing query
	coral::ICursor& cursorI = queryI->execute();
	//a little printout, then filling DQMSummary
	int nRowsI = 0;
	while( cursorI.next() ) {
	  //cursorI.currentRow().toOutputStream(std::cout) << std::endl;
	  ++nRowsI;
	  const coral::AttributeList& rowI = cursorI.currentRow();
	  runItem.m_lumisec = rowI["lumisec"].data<long long>();
	  //defining second query
	  coral::IQuery* queryII = schema.tableHandle(m_tableToRead).newQuery();
	  queryII->addToOutputList("subsystem");
	  queryII->addToOutputList("reportcontent");
	  //queryII->addToOutputList("type"); // when implemented in OMDS
	  queryII->addToOutputList("status");
	  std::string conditionII = "run= :n_run AND lumisec= :n_lumisec";
	  coral::AttributeList conditionDataII;
	  conditionDataII.extend<long long>( "n_run" );
	  //conditionDataII[0].setValue<long long>(r_number); 
	  conditionDataII[0].data<long long>() = r_number;
	  conditionDataII.extend<long long>( "n_lumisec" );
	  //conditionDataII[1].setValue<long long>(rowI["lumisec"].data<long long>()); 
	  conditionDataII[1].data<long long>() = rowI["lumisec"].data<long long>();
	  queryII->setCondition(conditionII, conditionDataII);
	  //performing query
	  coral::ICursor& cursorII = queryII->execute();
	  //a little printout, then filling DQMSummary
	  int nRowsII = 0;
	  while( cursorII.next() ) {
	    //cursorII.currentRow().toOutputStream(std::cout) << std::endl;
	    ++nRowsII;
	    const coral::AttributeList& rowII = cursorII.currentRow();
	    lumiItem.m_subsystem = rowII["subsystem"].data<std::string>();
	    lumiItem.m_reportcontent = rowII["reportcontent"].data<std::string>();
	    //lumiItem.m_type = rowII["type"].data<std::string>(); // when implemented in OMDS
	    lumiItem.m_type = "reportSummary";
	    lumiItem.m_status = rowII["status"].data<double>();
	    runItem.m_lumisummary.push_back(lumiItem);
	    std::cout << "DQMSummary::DQMSummary::RunItem::LumiItem filled" << std::endl;
	  }
	  std::cout << "Returned rows for lumisection query " << nRowsII << std::endl;
	  dqmSummary.m_summary.push_back(runItem);
	  std::cout << "DQMSummary::DQMSummary::RunItem filled" << std::endl;
	  delete queryII;
	}
	std::cout << "Returned rows for run number query " << nRowsI << std::endl;
	delete queryI;
      }
      else {
	runItem.m_lumisec = 0;
	lumiItem.m_subsystem = " ";
	lumiItem.m_reportcontent = " ";
	lumiItem.m_type = " ";
	lumiItem.m_status = -2;
	std::cout << "[lumisec (long long) : " << runItem.m_lumisec
		  << "], [subsystem (string) : " << lumiItem.m_subsystem
		  << "], [reportcontent (string) : " << lumiItem.m_reportcontent
		  << "], [type (string) : " << lumiItem.m_type
		  << "], [status (double) : " << lumiItem.m_status
		  << "]" << std::endl;
	runItem.m_lumisummary.push_back(lumiItem);
	dqmSummary.m_summary.push_back(runItem);
	std::cout << "No information in DQMSummary for run " 
		  << r_number <<std::endl;
      }
    }
    else 
      throw cms::Exception("UnconsistentData") << "What is wrong with you?" << std::endl;
    std::cout << "DQMSummary built" << std::endl;
    delete query;
    session->transaction().commit();
  } 
  catch (const std::exception& e) { 
    std::cout << "Exception: "<<e.what()<<std::endl; 
  }
  delete session;
  return dqmSummary;
}

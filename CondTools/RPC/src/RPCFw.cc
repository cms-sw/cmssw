 /* 
 *  See header file for a description of this class.
 *
 *  $Date: 2008/02/15 12:15:58 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "RPCFw.h"
#include "TimeConv.h"
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

RPCFw::RPCFw( const std::string& connectionString,
              const std::string& userName,
              const std::string& password):
  TestBase(),
  m_connectionString( connectionString ),
  m_userName( userName ),
  m_password( password )
{}


RPCFw::~RPCFw()
{}

void
RPCFw::run()
{
}


//----------------------------- I M O N ------------------------------------------------------------------------
std::vector<RPCdbData::Item> RPCFw::createIMON(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;
  //coral::TimeStamp thr = coral::TimeStamp(2007,11,2,03,11,00,00);
 
  coral::ISession* session = this->connect( m_connectionString,
                                            m_userName, m_password );
  session->transaction().start( true );
  coral::ISchema& schema = session->nominalSchema();
  int nRows = 0;
  std::cout << ">> creating IMON object..." << std::endl;
  coral::IQuery* queryI = schema.newQuery();
  queryI->addToTableList( "FWCAENCHANNEL" );
  queryI->addToOutputList( "FWCAENCHANNEL.DPID", "DPID" );
  queryI->addToOutputList( "FWCAENCHANNEL.CHANGE_DATE", "TSTAMP" );
  queryI->addToOutputList( "FWCAENCHANNEL.ACTUAL_IMON", "IMON" );
  std::string condI = "FWCAENCHANNEL.ACTUAL_IMON is not NULL AND ";

  std::string condition = "FWCAENCHANNEL.ACTUAL_IMON is not NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmax";
  coral::AttributeList conditionData;
  conditionData.extend<coral::TimeStamp>( "tmax" );
  queryI->setCondition( condition, conditionData );
  conditionData[0].data<coral::TimeStamp>() = thr;
  coral::ICursor& cursorI = queryI->execute();

  RPCdbData::Item Itemp;
  std::vector<RPCdbData::Item> imonarray;
  while ( cursorI.next() ) {
    const coral::AttributeList& row = cursorI.currentRow();
    float idoub = row["DPID"].data<float>();
    int id = static_cast<int>(idoub);
    float val = row["IMON"].data<float>();
    coral::TimeStamp ts =  row["TSTAMP"].data<coral::TimeStamp>();
    int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year()-2000);
    int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

    Itemp.dpid = id;
    Itemp.value = val;
    Itemp.day = ndate;
    Itemp.time = ntime;
    imonarray.push_back(Itemp);

    ++nRows;
  }
  

  std::cout << ">> Imon array --> size: " << imonarray.size() << " >> done." << std::endl;
  delete queryI;
  session->transaction().commit();
  delete session;
  return imonarray;
}



//------------------------------------------------------- V M O N ---------------------------------------------------
std::vector<RPCdbData::Item> RPCFw::createVMON(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;
  //coral::TimeStamp thr = this->lastValue();
  //coral::TimeStamp thr = coral::TimeStamp(2007,11,2,03,11,00,00);

  coral::ISession* session = this->connect( m_connectionString,
                                            m_userName, m_password );
  session->transaction().start( true );
  coral::ISchema& schema = session->nominalSchema();
  int nRows = 0;
  std::cout << ">> creating VMON object..." << std::endl;
  coral::IQuery* queryV = schema.newQuery();
  queryV->addToTableList( "FWCAENCHANNEL" );
  queryV->addToOutputList( "FWCAENCHANNEL.DPID", "DPID" );
  queryV->addToOutputList( "FWCAENCHANNEL.CHANGE_DATE", "TSTAMP" );
  queryV->addToOutputList( "FWCAENCHANNEL.ACTUAL_VMON", "VMON" );
  std::string condV = "FWCAENCHANNEL.ACTUAL_VMON is not NULL";
  
  std::string condition = "FWCAENCHANNEL.ACTUAL_VMON is not NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmax";
  coral::AttributeList conditionData;
  conditionData.extend<coral::TimeStamp>( "tmax" );
  queryV->setCondition( condition, conditionData );
  conditionData[0].data<coral::TimeStamp>() = thr;
  coral::ICursor& cursorV = queryV->execute();

  RPCdbData::Item Vtemp;
  std::vector<RPCdbData::Item> vmonarray;
  while ( cursorV.next() ) {
    const coral::AttributeList& row = cursorV.currentRow();
    float idoub = row["DPID"].data<float>();
    int id = static_cast<int>(idoub);
    float val = row["VMON"].data<float>();
    coral::TimeStamp ts =  row["TSTAMP"].data<coral::TimeStamp>();
    int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year()-2000);
    int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

    Vtemp.dpid = id;
    Vtemp.value = val;
    Vtemp.day = ndate;
    Vtemp.time = ntime;
    vmonarray.push_back(Vtemp);

    ++nRows;
  }
  std::cout << ">> Vmon array --> size: " << vmonarray.size() << " >> done." << std::endl;
  delete queryV;
  session->transaction().commit();
  delete session;
  return vmonarray;
}


//------------------------------ S T A T U S ---------------------------------------------------------------------
std::vector<RPCdbData::Item> RPCFw::createSTATUS(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;
  //coral::TimeStamp thr = this->lastValue();
  //coral::TimeStamp thr = coral::TimeStamp(2007,11,2,03,11,00,00);

  coral::ISession* session = this->connect( m_connectionString,
                                            m_userName, m_password );
  session->transaction().start( true );
  coral::ISchema& schema = session->nominalSchema();
  int nRows = 0;
  std::cout << ">> creating STATUS object..." << std::endl;
  coral::IQuery* queryS = schema.newQuery();
  queryS->addToTableList( "FWCAENCHANNEL" );
  queryS->addToOutputList( "FWCAENCHANNEL.DPID", "DPID" );
  queryS->addToOutputList( "FWCAENCHANNEL.CHANGE_DATE", "TSTAMP" );
  queryS->addToOutputList( "FWCAENCHANNEL.ACTUAL_STATUS", "STATUS" );
  std::string condS = "FWCAENCHANNEL.ACTUAL_STATUS is not NULL";

  std::string condition = "FWCAENCHANNEL.ACTUAL_STATUS is not NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmax";
  coral::AttributeList conditionData;
  conditionData.extend<coral::TimeStamp>( "tmax" );
  queryS->setCondition( condition, conditionData );
  conditionData[0].data<coral::TimeStamp>() = thr;
  coral::ICursor& cursorS = queryS->execute();

  RPCdbData::Item Stemp;
  std::vector<RPCdbData::Item> statusarray;
  while ( cursorS.next() ) {
    const coral::AttributeList& row = cursorS.currentRow();
    float idoub = row["DPID"].data<float>();
    int id = static_cast<int>(idoub);
    float val = row["STATUS"].data<float>();
    coral::TimeStamp ts =  row["TSTAMP"].data<coral::TimeStamp>();
    int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year()-2000);
    int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

    Stemp.dpid = id;
    Stemp.value = val;
    Stemp.day = ndate;
    Stemp.time = ntime;
    statusarray.push_back(Stemp);

    ++nRows;
  }
  std::cout << ">> Staus array --> size: " << statusarray.size() << " >> done." << std::endl << std::endl << std::endl;
  
  delete queryS;
  session->transaction().commit();
  delete session;

  return statusarray;

}

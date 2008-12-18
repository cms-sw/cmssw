 /* 
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/25 15:00:14 $
 *  $Revision: 1.9 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondTools/RPC/interface/RPCFw.h"
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
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>

RPCFw::RPCFw( const std::string& connectionString,
              const std::string& userName,
              const std::string& password):
  RPCDBCom(),
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
std::vector<RPCObCond::Item> RPCFw::createIMON(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;
 
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

  RPCObCond::Item Itemp;
  std::vector<RPCObCond::Item> imonarray;
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
std::vector<RPCObCond::Item> RPCFw::createVMON(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;

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

  RPCObCond::Item Vtemp;
  std::vector<RPCObCond::Item> vmonarray;
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
std::vector<RPCObCond::Item> RPCFw::createSTATUS(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;

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

  RPCObCond::Item Stemp;
  std::vector<RPCObCond::Item> statusarray;
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



//------------------------------ G A S ---------------------------------------------------------------------
std::vector<RPCObGas::Item> RPCFw::createGAS(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;

  coral::ISession* session = this->connect( m_connectionString,
                                            m_userName, m_password );
  session->transaction().start( true );
  coral::ISchema& schema = session->nominalSchema();
  int nRows = 0;
  std::cout << ">> creating GAS object..." << std::endl;
  coral::IQuery* queryS = schema.newQuery();
  queryS->addToTableList( "RPCGASCHANNEL" );
  queryS->addToOutputList( "RPCGASCHANNEL.DPID", "DPID" );
  queryS->addToOutputList( "RPCGASCHANNEL.CHANGE_DATE", "TSTAMP" );
  queryS->addToOutputList( "RPCGASCHANNEL.FLOWIN", "FLOWIN" );
  queryS->addToOutputList( "RPCGASCHANNEL.FLOWOUT", "FLOWOUT" );
  std::string condS = "RPCGASCHANNEL.FLOWIN is not NULL";

  std::string condition = "RPCGASCHANNEL.FLOWIN is not NULL AND RPCGASCHANNEL.CHANGE_DATE >:tmax";
  coral::AttributeList conditionData;
  conditionData.extend<coral::TimeStamp>( "tmax" );
  queryS->setCondition( condition, conditionData );
  conditionData[0].data<coral::TimeStamp>() = thr;
  coral::ICursor& cursorS = queryS->execute();

  RPCObGas::Item gastemp;
  std::vector<RPCObGas::Item> gasarray;
  while ( cursorS.next() ) {
    const coral::AttributeList& row = cursorS.currentRow();
    float idoub = row["DPID"].data<float>();
    int id = static_cast<int>(idoub);
    float valin = row["FLOWIN"].data<float>();
    float valout = row["FLOWOUT"].data<float>();
    coral::TimeStamp ts =  row["TSTAMP"].data<coral::TimeStamp>();
    int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year()-2000);
    int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

    gastemp.dpid = id;
    gastemp.flowin = valin;
    gastemp.flowout = valout;
    gastemp.day = ndate;
    gastemp.time = ntime;
    gasarray.push_back(gastemp);

    ++nRows;
  }
  std::cout << ">> Gas array --> size: " << gasarray.size() << " >> done." << std::endl << std::endl << std::endl;

  delete queryS;
  session->transaction().commit();
  delete session;

  return gasarray;

}



//------------------------------ T E M P E R A T U R E ---------------------------------------------------------------------
std::vector<RPCObCond::Item> RPCFw::createT(int from)
{
  thr = UTtoT(from);
  std::cout <<">> Processing since: "<<thr.day()<<"/"<<thr.month()<<"/"<<thr.year()<<" "<<thr.hour()<<":"<<thr.minute()<<"."<<thr.second()<< std::endl;

  coral::ISession* session = this->connect( m_connectionString,
                                            m_userName, m_password );
  session->transaction().start( true );
  coral::ISchema& schema = session->nominalSchema();
  int nRows = 0;
  std::cout << ">> creating TEMPERATURE object..." << std::endl;
  coral::IQuery* queryS = schema.newQuery();
  queryS->addToTableList( "FWCAENCHANNELADC" );
  queryS->addToOutputList( "FWCAENCHANNELADC.DPID", "DPID" );
  queryS->addToOutputList( "FWCAENCHANNELADC.CHANGE_DATE", "TSTAMP" );
  queryS->addToOutputList( "FWCAENCHANNELADC.ACTUAL_TEMPERATURE", "TEMPERATURE" );
  std::string condS = "FWCAENCHANNELADC.ACTUAL_TEMPERATURE is not NULL";

  std::string condition = "FWCAENCHANNELADC.ACTUAL_TEMPERATURE is not NULL AND FWCAENCHANNELADC.CHANGE_DATE >:tmax";
  coral::AttributeList conditionData;
  conditionData.extend<coral::TimeStamp>( "tmax" );
  queryS->setCondition( condition, conditionData );
  conditionData[0].data<coral::TimeStamp>() = thr;
  coral::ICursor& cursorS = queryS->execute();

  RPCObCond::Item Ttemp;
  std::vector<RPCObCond::Item> temparray;
  while ( cursorS.next() ) {
    const coral::AttributeList& row = cursorS.currentRow();
    float idoub = row["DPID"].data<float>();
    int id = static_cast<int>(idoub);
    float val = row["TEMPERATURE"].data<float>();
    coral::TimeStamp ts =  row["TSTAMP"].data<coral::TimeStamp>();
    int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year()-2000);
    int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

    Ttemp.dpid = id;
    Ttemp.value = val;
    Ttemp.day = ndate;
    Ttemp.time = ntime;
    temparray.push_back(Ttemp);

    ++nRows;
  }
  std::cout << ">> Temperature array --> size: " << temparray.size() << " >> done." << std::endl << std::endl << std::endl;

  delete queryS;
  session->transaction().commit();
  delete session;

  return temparray;

}





//----------------------------------------------------------------------------------------------
coral::TimeStamp RPCFw::UTtoT(int utime) 
{
  int yea = static_cast<int>(trunc(utime/31536000) + 1970);
  int yes = (yea-1970)*31536000;
  int cony = (yea-1972)%4;
  if (cony == 0) yes = yes + (yea-1972)/4*86400; 
  else yes = yes +  static_cast<int>(trunc((yea-1972)/4))*86400;
  int day = static_cast<int>(trunc((utime - yes)/86400));
  int rest = static_cast<int>(utime - yes - day*86400);
  int mon = 0;
  // BISESTILE YEAR
  if (cony == 0) {
   day = day + 1; 
   if (day < 32){
      mon = 1;
      day = day - 0;
    }
    if (day >= 32 && day < 61){
      mon = 2;
      day = day - 31;
    }
    if (day >= 61 && day < 92){
      mon = 3;
      day = day - 60;
    }
    if (day >= 92 && day < 122){
      mon = 4;
      day = day - 91;
    }
    if (day >= 122 && day < 153){
      mon = 5;
      day = day - 121;
    }
    if (day >= 153 && day < 183){
      mon = 6;
      day = day - 152;
    }
    if (day >= 183 && day < 214){
      mon = 7;
      day = day - 182;
    }
    if (day >= 214 && day < 245){
      mon = 8;
      day = day - 213;
    }
    if (day >= 245 && day < 275){
      mon = 9;
      day = day - 244;
    }
    if (day >= 275 && day < 306){
      mon = 10;
      day = day - 274;
    }
    if (day >= 306 && day < 336){
      mon = 11;
      day = day - 305;
    }
    if (day >= 336){
      mon = 12;
      day = day - 335;
    }
  }
  // NOT BISESTILE YEAR
  else {
    if (day < 32){
      mon = 1;   
      day = day - 0;
    }
    if (day >= 32 && day < 60){
      mon = 2;
      day = day - 31;
    }
    if (day >= 60 && day < 91){
      mon = 3;
      day = day - 59;
    }
    if (day >= 91 && day < 121){
      mon = 4;
      day = day - 90;
    }
    if (day >= 121 && day < 152){
      mon = 5;
      day = day - 120;
    }
    if (day >= 152 && day < 182){
      mon = 6;
      day = day - 151;
    }
    if (day >= 182 && day < 213){
      mon = 7;
      day = day - 181;
    }
    if (day >= 213 && day < 244){
      mon = 8;
      day = day - 212;
    }
    if (day >= 244 && day < 274){
      mon = 9;
      day = day - 243;
    }
    if (day >= 274 && day < 305){
      mon = 10;
      day = day - 273;
    }
    if (day >= 305 && day < 335){
      mon = 11;
      day = day - 304;
    }
    if (day >= 335){
      mon = 12;
      day = day - 334;
    }
  }
  
  int hou = static_cast<int>(trunc(rest/3600)); 
  rest = rest - hou*3600;
  int min = static_cast<int>(trunc(rest/60));
  rest = rest - min*60;
  int sec = rest; 
  int nan = 0;

  //  std::cout <<">> Processing since: "<<day<<"/"<<mon<<"/"<<yea<<" "<<hou<<":"<<min<<"."<<sec<< std::endl;

  coral::TimeStamp Tthr;  

  Tthr = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
  return Tthr;
}


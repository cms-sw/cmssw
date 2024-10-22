
/* 
 *  See header file for a description of this class.
 *
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
#include <cmath>
#include <iostream>
#include <sstream>
#include <ctime>
#include "CondFormats/RPCObjects/interface/RPCObFebmap.h"

RPCFw::RPCFw(const std::string& connectionString, const std::string& userName, const std::string& password)
    : RPCDBCom(), m_connectionString(connectionString), m_userName(userName), m_password(password) {}

RPCFw::~RPCFw() {}

void RPCFw::run() {}

//----------------------------- I M O N ------------------------------------------------------------------------
std::vector<RPCObImon::I_Item> RPCFw::createIMON(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  coral::IQuery* queryI = schema.newQuery();
  queryI->addToTableList("FWCAENCHANNEL");
  queryI->addToOutputList("FWCAENCHANNEL.DPID", "DPID");
  queryI->addToOutputList("FWCAENCHANNEL.CHANGE_DATE", "TSTAMP");
  queryI->addToOutputList("FWCAENCHANNEL.ACTUAL_IMON", "IMON");

  RPCObImon::I_Item Itemp;
  std::vector<RPCObImon::I_Item> imonarray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << ">> creating IMON object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string condition = "FWCAENCHANNEL.ACTUAL_IMON IS NOT NULL AND CHANGE_DATE >:tmin AND CHANGE_DATE <:tmax";
    queryI->setCondition(condition, conditionData);
    coral::ICursor& cursorI = queryI->execute();
    while (cursorI.next()) {
      const coral::AttributeList& row = cursorI.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["IMON"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.value = val;
      Itemp.day = ndate;
      Itemp.time = ntime;
      imonarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> creating IMON object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string condition = "FWCAENCHANNEL.ACTUAL_IMON IS NOT NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmin";
    queryI->setCondition(condition, conditionData);
    coral::ICursor& cursorI = queryI->execute();
    while (cursorI.next()) {
      const coral::AttributeList& row = cursorI.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["IMON"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      if (isMajor(ts, tlast))
        tlast = ts;
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.value = val;
      Itemp.day = ndate;
      Itemp.time = ntime;
      imonarray.push_back(Itemp);
    }
  }
  N_IOV = TtoUT(tlast);
  if (till > since)
    N_IOV = till;
  std::cout << ">> Imon array --> size: " << imonarray.size() << " >> done." << std::endl;
  delete queryI;
  session->transaction().commit();
  delete session;
  return imonarray;
}

//------------------------------------------------------- V M O N ---------------------------------------------------
std::vector<RPCObVmon::V_Item> RPCFw::createVMON(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  coral::IQuery* queryV = schema.newQuery();
  queryV->addToTableList("FWCAENCHANNEL");
  queryV->addToOutputList("FWCAENCHANNEL.DPID", "DPID");
  queryV->addToOutputList("FWCAENCHANNEL.CHANGE_DATE", "TSTAMP");
  queryV->addToOutputList("FWCAENCHANNEL.ACTUAL_VMON", "VMON");
  RPCObVmon::V_Item Vtemp;
  std::vector<RPCObVmon::V_Item> vmonarray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << ">> creating VMON object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string condition =
        " NOT FWCAENCHANNEL.ACTUAL_VMON IS NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmin AND FWCAENCHANNEL.CHANGE_DATE "
        "<:tmax";
    queryV->setCondition(condition, conditionData);
    coral::ICursor& cursorV = queryV->execute();
    while (cursorV.next()) {
      const coral::AttributeList& row = cursorV.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["VMON"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      if (isMajor(ts, tlast))
        tlast = ts;
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Vtemp.dpid = id;
      Vtemp.value = val;
      Vtemp.day = ndate;
      Vtemp.time = ntime;
      vmonarray.push_back(Vtemp);
    }
  } else {
    std::cout << ">> creating VMON object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string condition = " NOT FWCAENCHANNEL.ACTUAL_VMON IS NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmin";
    queryV->setCondition(condition, conditionData);
    coral::ICursor& cursorV = queryV->execute();
    while (cursorV.next()) {
      const coral::AttributeList& row = cursorV.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["VMON"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      if (isMajor(ts, tlast))
        tlast = ts;
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Vtemp.dpid = id;
      Vtemp.value = val;
      Vtemp.day = ndate;
      Vtemp.time = ntime;
      vmonarray.push_back(Vtemp);
    }
  }

  std::cout << ">> Vmon array --> size: " << vmonarray.size() << " >> done." << std::endl;
  delete queryV;
  session->transaction().commit();
  delete session;
  return vmonarray;
}

//------------------------------ S T A T U S ---------------------------------------------------------------------
std::vector<RPCObStatus::S_Item> RPCFw::createSTATUS(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  std::cout << ">> creating STATUS object..." << std::endl;
  coral::IQuery* queryS = schema.newQuery();
  queryS->addToTableList("FWCAENCHANNEL");
  queryS->addToOutputList("FWCAENCHANNEL.DPID", "DPID");
  queryS->addToOutputList("FWCAENCHANNEL.CHANGE_DATE", "TSTAMP");
  queryS->addToOutputList("FWCAENCHANNEL.ACTUAL_STATUS", "STATUS");
  RPCObStatus::S_Item Stemp;
  std::vector<RPCObStatus::S_Item> statusarray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << ">> creating STATUS object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string condition =
        " NOT FWCAENCHANNEL.ACTUAL_STATUS IS NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmin AND FWCAENCHANNEL.CHANGE_DATE "
        "<:tmax";
    queryS->setCondition(condition, conditionData);
    coral::ICursor& cursorS = queryS->execute();
    while (cursorS.next()) {
      const coral::AttributeList& row = cursorS.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["STATUS"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Stemp.dpid = id;
      Stemp.value = val;
      Stemp.day = ndate;
      Stemp.time = ntime;
      statusarray.push_back(Stemp);
    }
  } else {
    std::cout << ">> creating STATUS object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string condition = " NOT FWCAENCHANNEL.ACTUAL_STATUS IS NULL AND FWCAENCHANNEL.CHANGE_DATE >:tmin";
    queryS->setCondition(condition, conditionData);
    coral::ICursor& cursorS = queryS->execute();
    while (cursorS.next()) {
      const coral::AttributeList& row = cursorS.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["STATUS"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Stemp.dpid = id;
      Stemp.value = val;
      Stemp.day = ndate;
      Stemp.time = ntime;
      statusarray.push_back(Stemp);
    }
  }
  std::cout << ">> Staus array --> size: " << statusarray.size() << " >> done." << std::endl << std::endl << std::endl;

  delete queryS;
  session->transaction().commit();
  delete session;

  return statusarray;
}

//------------------------------ G A S ---------------------------------------------------------------------
std::vector<RPCObGas::Item> RPCFw::createGAS(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();

  coral::TimeStamp tlast = tMIN;

  // FLOWIN
  coral::IQuery* querySIN = schema.newQuery();
  querySIN->addToTableList("RPCGASCHANNEL");
  querySIN->addToOutputList("RPCGASCHANNEL.DPID", "DPID");
  querySIN->addToOutputList("RPCGASCHANNEL.CHANGE_DATE", "TSTAMP");
  querySIN->addToOutputList("RPCGASCHANNEL.FLOWIN", "FLOWIN");

  // FLOWOUT
  coral::IQuery* querySOUT = schema.newQuery();
  querySOUT->addToTableList("RPCGASCHANNEL");
  querySOUT->addToOutputList("RPCGASCHANNEL.DPID", "DPID");
  querySOUT->addToOutputList("RPCGASCHANNEL.CHANGE_DATE", "TSTAMP");
  querySOUT->addToOutputList("RPCGASCHANNEL.FLOWOUT", "FLOWOUT");

  RPCObGas::Item gastemp;
  std::vector<RPCObGas::Item> gasarray;

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << ">> creating GAS object..." << std::endl;
    std::cout << ">> processing FLOWIN..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionIN = "RPCGASCHANNEL.FLOWIN IS NOT NULL AND CHANGE_DATE >:tmin AND CHANGE_DATE <:tmax";

    querySIN->setCondition(conditionIN, conditionData);
    coral::ICursor& cursorSIN = querySIN->execute();
    while (cursorSIN.next()) {
      gastemp.dpid = 0;
      gastemp.flowin = 0;
      gastemp.flowout = 0;
      gastemp.day = 0;
      gastemp.time = 0;
      const coral::AttributeList& row = cursorSIN.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["FLOWIN"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      gastemp.dpid = id;
      gastemp.flowin = val;
      gastemp.day = ndate;
      gastemp.time = ntime;
      gasarray.push_back(gastemp);
    }
  } else {
    std::cout << ">> creating GAS object..." << std::endl;
    std::cout << ">> processing FLOWIN..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionIN = "RPCGASCHANNEL.FLOWIN IS NOT NULL AND CHANGE_DATE >:tmin";
    std::cout << "processing FLOWIN..." << std::endl;
    querySIN->setCondition(conditionIN, conditionData);
    coral::ICursor& cursorSIN = querySIN->execute();
    while (cursorSIN.next()) {
      gastemp.dpid = 0;
      gastemp.flowin = 0;
      gastemp.flowout = 0;
      gastemp.day = 0;
      gastemp.time = 0;
      const coral::AttributeList& row = cursorSIN.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["FLOWIN"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      gastemp.dpid = id;
      gastemp.flowin = val;
      gastemp.day = ndate;
      gastemp.time = ntime;
      gasarray.push_back(gastemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    //    std::cout <<">> Processing till: "<<tMAX.day()<<"/"<<tMAX.month()<<"/"<<tMAX.year()<<" "<<tMAX.hour()<<":"<<tMAX.minute()<<"."<<tMAX.second()<< std::endl;
    std::cout << ">> processing FLOWOUT..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionOUT = "RPCGASCHANNEL.FLOWOUT IS NOT NULL AND CHANGE_DATE >:tmin AND CHANGE_DATE <:tmax";

    querySOUT->setCondition(conditionOUT, conditionData);
    coral::ICursor& cursorSOUT = querySOUT->execute();
    while (cursorSOUT.next()) {
      gastemp.dpid = 0;
      gastemp.flowin = 0;
      gastemp.flowout = 0;
      gastemp.day = 0;
      gastemp.time = 0;
      const coral::AttributeList& row = cursorSOUT.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["FLOWOUT"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      gastemp.dpid = id;
      gastemp.flowout = val;
      gastemp.day = ndate;
      gastemp.time = ntime;
      gasarray.push_back(gastemp);
    }
  } else {
    std::cout << ">> processing FLOWOUT..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionOUT = "RPCGASCHANNEL.FLOWOUT IS NOT NULL AND CHANGE_DATE >:tmin";
    querySOUT->setCondition(conditionOUT, conditionData);
    coral::ICursor& cursorSOUT = querySOUT->execute();
    while (cursorSOUT.next()) {
      gastemp.dpid = 0;
      gastemp.flowin = 0;
      gastemp.flowout = 0;
      gastemp.day = 0;
      gastemp.time = 0;
      const coral::AttributeList& row = cursorSOUT.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["FLOWOUT"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      gastemp.dpid = id;
      gastemp.flowout = val;
      gastemp.day = ndate;
      gastemp.time = ntime;
      gasarray.push_back(gastemp);
    }
  }

  N_IOV = TtoUT(tlast);
  if (till > since)
    N_IOV = till;
  std::cout << ">> Gas array --> size: " << gasarray.size() << " >> done." << std::endl << std::endl << std::endl;

  delete querySIN;
  delete querySOUT;
  session->transaction().commit();
  delete session;

  return gasarray;
}

//------------------------------ T E M P E R A T U R E ---------------------------------------------------------------------
std::vector<RPCObTemp::T_Item> RPCFw::createT(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  coral::IQuery* queryS = schema.newQuery();
  queryS->addToTableList("FWCAENCHANNELADC");
  queryS->addToOutputList("FWCAENCHANNELADC.DPID", "DPID");
  queryS->addToOutputList("FWCAENCHANNELADC.CHANGE_DATE", "TSTAMP");
  queryS->addToOutputList("FWCAENCHANNELADC.ACTUAL_TEMPERATURE", "TEMPERATURE");
  RPCObTemp::T_Item Ttemp;
  std::vector<RPCObTemp::T_Item> temparray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << ">> creating TEMPERATURE object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string condition =
        "FWCAENCHANNELADC.ACTUAL_TEMPERATURE IS NOT NULL AND CHANGE_DATE >:tmin AND CHANGE_DATE <:tmax";
    queryS->setCondition(condition, conditionData);
    coral::ICursor& cursorS = queryS->execute();
    while (cursorS.next()) {
      const coral::AttributeList& row = cursorS.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["TEMPERATURE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

      Ttemp.dpid = id;
      Ttemp.value = val;
      Ttemp.day = ndate;
      Ttemp.time = ntime;
      temparray.push_back(Ttemp);
    }
  } else {
    std::cout << ">> creating TEMPERATURE object..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string condition = "FWCAENCHANNELADC.ACTUAL_TEMPERATURE IS NOT NULL AND CHANGE_DATE >:tmin";
    queryS->setCondition(condition, conditionData);
    coral::ICursor& cursorS = queryS->execute();
    while (cursorS.next()) {
      const coral::AttributeList& row = cursorS.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float val = row["TEMPERATURE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();

      Ttemp.dpid = id;
      Ttemp.value = val;
      Ttemp.day = ndate;
      Ttemp.time = ntime;
      temparray.push_back(Ttemp);
    }
  }
  N_IOV = TtoUT(tlast);
  if (till > since)
    N_IOV = till;
  std::cout << ">> Temperature array --> size: " << temparray.size() << " >> done." << std::endl
            << std::endl
            << std::endl;

  delete queryS;
  session->transaction().commit();
  delete session;

  return temparray;
}

//----------------------------- I D   M A P ------------------------------------------------------------------------
std::vector<RPCObPVSSmap::Item> RPCFw::createIDMAP() {
  //  float tMINi = 0;
  std::cout << ">> Processing data..." << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  std::cout << ">> creating IDMAP object..." << std::endl;
  coral::IQuery* queryM = schema.newQuery();
  queryM->addToTableList("RPCPVSSDETID");
  queryM->addToOutputList("RPCPVSSDETID.SINCE", "SINCE");
  queryM->addToOutputList("RPCPVSSDETID.PVSS_ID", "PVSS_ID");
  queryM->addToOutputList("RPCPVSSDETID.REGION", "REGION");
  queryM->addToOutputList("RPCPVSSDETID.RING", "RING");
  queryM->addToOutputList("RPCPVSSDETID.STATION", "STATION");
  queryM->addToOutputList("RPCPVSSDETID.SECTOR", "SECTOR");
  queryM->addToOutputList("RPCPVSSDETID.LAYER", "LAYER");
  queryM->addToOutputList("RPCPVSSDETID.SUBSECTOR", "SUBSECTOR");
  queryM->addToOutputList("RPCPVSSDETID.SUPPLYTYPE", "SUPPLYTYPE");

  std::string condM = "RPCPVSSDETID.PVSS_ID is not NULL";

  //  queryM->setCondition(condM);
  coral::ICursor& cursorM = queryM->execute();

  RPCObPVSSmap::Item Itemp;
  std::vector<RPCObPVSSmap::Item> idmaparray;
  while (cursorM.next()) {
    const coral::AttributeList& row = cursorM.currentRow();
    int id = row["PVSS_ID"].data<int>();
    std::string reg_s = row["REGION"].data<std::string>();
    std::string rin_s = row["RING"].data<std::string>();
    std::string sta_s = row["STATION"].data<std::string>();
    std::string sec_s = row["SECTOR"].data<std::string>();
    std::string lay_s = row["LAYER"].data<std::string>();
    std::string sub_s = row["SUBSECTOR"].data<std::string>();
    std::string sup_s = row["SUPPLYTYPE"].data<std::string>();

    int reg = atoi(reg_s.c_str());
    int rin = atoi(rin_s.c_str());
    int sta = atoi(sta_s.c_str());
    int sec = atoi(sec_s.c_str());
    int lay = atoi(lay_s.c_str());
    int sub = atoi(sub_s.c_str());
    int sup = 5;

    if (sup_s == "HV")
      sup = 0;
    if (sup_s == "LVA")
      sup = 1;
    if (sup_s == "LVD")
      sup = 2;
    if (sup_s == "LB")
      sup = 3;
    if (sup_s == "T")
      sup = 4;

    coral::TimeStamp ts = row["SINCE"].data<coral::TimeStamp>();

    std::ostringstream dday;
    std::ostringstream dmon;
    std::ostringstream dyea;
    dday << ts.day();
    dmon << ts.month();
    dyea << ts.year();
    std::string date = dday.str() + "-" + dmon.str() + "-" + dyea.str();

    Itemp.since = atoi(date.c_str());
    Itemp.dpid = id;
    Itemp.region = reg;
    Itemp.ring = rin;
    Itemp.station = sta;
    Itemp.sector = sec;
    Itemp.layer = lay;
    Itemp.subsector = sub;
    Itemp.suptype = sup;

    idmaparray.push_back(Itemp);
  }

  std::cout << ">> IDMAP array --> size: " << idmaparray.size() << " >> done." << std::endl;
  delete queryM;
  session->transaction().commit();
  delete session;
  return idmaparray;
}

//----------------------------- F E B ------------------------------------------------------------------------
std::vector<RPCObFebmap::Feb_Item> RPCFw::createFEB(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  std::cout << ">> creating FEB object..." << std::endl;

  // VTH1
  coral::IQuery* queryFVTH1 = schema.newQuery();
  queryFVTH1->addToTableList("RPCFEB");
  queryFVTH1->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVTH1->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVTH1->addToOutputList("RPCFEB.VTH1", "VTH1");

  // VTH2
  coral::IQuery* queryFVTH2 = schema.newQuery();
  queryFVTH2->addToTableList("RPCFEB");
  queryFVTH2->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVTH2->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVTH2->addToOutputList("RPCFEB.VTH2", "VTH2");

  // VTH3
  coral::IQuery* queryFVTH3 = schema.newQuery();
  queryFVTH3->addToTableList("RPCFEB");
  queryFVTH3->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVTH3->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVTH3->addToOutputList("RPCFEB.VTH3", "VTH3");

  // VTH4
  coral::IQuery* queryFVTH4 = schema.newQuery();
  queryFVTH4->addToTableList("RPCFEB");
  queryFVTH4->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVTH4->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVTH4->addToOutputList("RPCFEB.VTH4", "VTH4");

  // VMON1
  coral::IQuery* queryFVMON1 = schema.newQuery();
  queryFVMON1->addToTableList("RPCFEB");
  queryFVMON1->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVMON1->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVMON1->addToOutputList("RPCFEB.VMON1", "VMON1");

  // VMON2
  coral::IQuery* queryFVMON2 = schema.newQuery();
  queryFVMON2->addToTableList("RPCFEB");
  queryFVMON2->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVMON2->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVMON2->addToOutputList("RPCFEB.VMON2", "VMON2");

  // VMON3
  coral::IQuery* queryFVMON3 = schema.newQuery();
  queryFVMON3->addToTableList("RPCFEB");
  queryFVMON3->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVMON3->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVMON3->addToOutputList("RPCFEB.VMON3", "VMON3");

  // VMON4
  coral::IQuery* queryFVMON4 = schema.newQuery();
  queryFVMON4->addToTableList("RPCFEB");
  queryFVMON4->addToOutputList("RPCFEB.DPID", "DPID");
  queryFVMON4->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFVMON4->addToOutputList("RPCFEB.VMON4", "VMON4");

  // TEMP1
  coral::IQuery* queryFTEMP1 = schema.newQuery();
  queryFTEMP1->addToTableList("RPCFEB");
  queryFTEMP1->addToOutputList("RPCFEB.DPID", "DPID");
  queryFTEMP1->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFTEMP1->addToOutputList("RPCFEB.TEMPERATURE1", "TEMP1");

  // TEMP2
  coral::IQuery* queryFTEMP2 = schema.newQuery();
  queryFTEMP2->addToTableList("RPCFEB");
  queryFTEMP2->addToOutputList("RPCFEB.DPID", "DPID");
  queryFTEMP2->addToOutputList("RPCFEB.CHANGE_DATE", "TSTAMP");
  queryFTEMP2->addToOutputList("RPCFEB.TEMPERATURE2", "TEMP2");

  RPCObFebmap::Feb_Item Itemp;
  std::vector<RPCObFebmap::Feb_Item> febarray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << "Processing VTH1..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVTH1 = "RPCFEB.VTH1 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVTH1->setCondition(conditionVTH1, conditionData);
    coral::ICursor& cursorFVTH1 = queryFVTH1->execute();
    while (cursorFVTH1.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH1.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth1 = row["VTH1"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr1 = vth1;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VTH1..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVTH1 = "RPCFEB.VTH1 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVTH1->setCondition(conditionVTH1, conditionData);
    coral::ICursor& cursorFVTH1 = queryFVTH1->execute();
    while (cursorFVTH1.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH1.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth1 = row["VTH1"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr1 = vth1;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VTH2..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVTH2 = "RPCFEB.VTH2 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVTH2->setCondition(conditionVTH2, conditionData);
    coral::ICursor& cursorFVTH2 = queryFVTH2->execute();
    while (cursorFVTH2.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH2.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth2 = row["VTH2"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr2 = vth2;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VTH2..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVTH2 = "RPCFEB.VTH2 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVTH2->setCondition(conditionVTH2, conditionData);
    coral::ICursor& cursorFVTH2 = queryFVTH2->execute();
    while (cursorFVTH2.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH2.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth2 = row["VTH2"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr2 = vth2;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VTH3..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVTH3 = "RPCFEB.VTH3 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVTH3->setCondition(conditionVTH3, conditionData);
    coral::ICursor& cursorFVTH3 = queryFVTH3->execute();
    while (cursorFVTH3.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH3.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth3 = row["VTH3"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr3 = vth3;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VTH3..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVTH3 = "RPCFEB.VTH3 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVTH3->setCondition(conditionVTH3, conditionData);
    coral::ICursor& cursorFVTH3 = queryFVTH3->execute();
    while (cursorFVTH3.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH3.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth3 = row["VTH3"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr3 = vth3;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VTH4..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVTH4 = "RPCFEB.VTH4 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVTH4->setCondition(conditionVTH4, conditionData);
    coral::ICursor& cursorFVTH4 = queryFVTH4->execute();
    while (cursorFVTH4.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH4.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth4 = row["VTH4"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr4 = vth4;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VTH4..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVTH4 = "RPCFEB.VTH4 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVTH4->setCondition(conditionVTH4, conditionData);
    coral::ICursor& cursorFVTH4 = queryFVTH4->execute();
    while (cursorFVTH4.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVTH4.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vth4 = row["VTH4"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.thr4 = vth4;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VMON1..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVMON1 = "RPCFEB.VMON1 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVMON1->setCondition(conditionVMON1, conditionData);
    coral::ICursor& cursorFVMON1 = queryFVMON1->execute();
    while (cursorFVMON1.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON1.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon1 = row["VMON1"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon1 = vmon1;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VMON1..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVMON1 = "RPCFEB.VMON1 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVMON1->setCondition(conditionVMON1, conditionData);
    coral::ICursor& cursorFVMON1 = queryFVMON1->execute();
    while (cursorFVMON1.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON1.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon1 = row["VMON1"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon1 = vmon1;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VMON2..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVMON2 = "RPCFEB.VMON2 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVMON2->setCondition(conditionVMON2, conditionData);
    coral::ICursor& cursorFVMON2 = queryFVMON2->execute();
    while (cursorFVMON2.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON2.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon2 = row["VMON2"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon2 = vmon2;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VMON2..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVMON2 = "RPCFEB.VMON2 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVMON2->setCondition(conditionVMON2, conditionData);
    coral::ICursor& cursorFVMON2 = queryFVMON2->execute();
    while (cursorFVMON2.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON2.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon2 = row["VMON2"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon2 = vmon2;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VMON3..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVMON3 = "RPCFEB.VMON3 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVMON3->setCondition(conditionVMON3, conditionData);
    coral::ICursor& cursorFVMON3 = queryFVMON3->execute();
    while (cursorFVMON3.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON3.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon3 = row["VMON3"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon3 = vmon3;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VMON3..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVMON3 = "RPCFEB.VMON3 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVMON3->setCondition(conditionVMON3, conditionData);
    coral::ICursor& cursorFVMON3 = queryFVMON3->execute();
    while (cursorFVMON3.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON3.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon3 = row["VMON3"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon3 = vmon3;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing VMON4..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionVMON4 = "RPCFEB.VMON4 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFVMON4->setCondition(conditionVMON4, conditionData);
    coral::ICursor& cursorFVMON4 = queryFVMON4->execute();
    while (cursorFVMON4.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON4.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon4 = row["VMON4"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon4 = vmon4;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing VMON4..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionVMON4 = "RPCFEB.VMON4 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFVMON4->setCondition(conditionVMON4, conditionData);
    coral::ICursor& cursorFVMON4 = queryFVMON4->execute();
    while (cursorFVMON4.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFVMON4.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float vmon4 = row["VMON4"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.vmon4 = vmon4;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing TEMP1..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionTEMP1 =
        "RPCFEB.TEMPERATURE1 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFTEMP1->setCondition(conditionTEMP1, conditionData);
    coral::ICursor& cursorFTEMP1 = queryFTEMP1->execute();
    while (cursorFTEMP1.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFTEMP1.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float temp1 = row["TEMP1"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.temp1 = temp1;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing TEMP1..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionTEMP1 = "RPCFEB.TEMPERATURE1 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFTEMP1->setCondition(conditionTEMP1, conditionData);
    coral::ICursor& cursorFTEMP1 = queryFTEMP1->execute();
    while (cursorFTEMP1.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFTEMP1.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float temp1 = row["TEMP1"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.temp1 = temp1;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << "Processing TEMP2..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionTEMP2 =
        "RPCFEB.TEMPERATURE2 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin AND RPCFEB.CHANGE_DATE <:tmax";
    queryFTEMP2->setCondition(conditionTEMP2, conditionData);
    coral::ICursor& cursorFTEMP2 = queryFTEMP2->execute();
    while (cursorFTEMP2.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFTEMP2.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float temp2 = row["TEMP2"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.temp2 = temp2;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  } else {
    std::cout << ">> Processing TEMP2..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    std::string conditionTEMP2 = "RPCFEB.TEMPERATURE2 IS NOT NULL AND RPCFEB.CHANGE_DATE >:tmin";
    queryFTEMP2->setCondition(conditionTEMP2, conditionData);
    coral::ICursor& cursorFTEMP2 = queryFTEMP2->execute();
    while (cursorFTEMP2.next()) {
      Itemp.thr1 = 0;
      Itemp.thr2 = 0;
      Itemp.thr3 = 0;
      Itemp.thr4 = 0;
      Itemp.vmon1 = 0;
      Itemp.vmon2 = 0;
      Itemp.vmon3 = 0;
      Itemp.vmon4 = 0;
      Itemp.temp1 = 0;
      Itemp.temp2 = 0;
      Itemp.noise1 = 0;
      Itemp.noise2 = 0;
      Itemp.noise3 = 0;
      Itemp.noise4 = 0;
      const coral::AttributeList& row = cursorFTEMP2.currentRow();
      float idoub = row["DPID"].data<float>();
      int id = static_cast<int>(idoub);
      float temp2 = row["TEMP2"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      int ndate = (ts.day() * 10000) + (ts.month() * 100) + (ts.year() - 2000);
      int ntime = (ts.hour() * 10000) + (ts.minute() * 100) + ts.second();
      Itemp.dpid = id;
      Itemp.temp2 = temp2;
      Itemp.day = ndate;
      Itemp.time = ntime;
      febarray.push_back(Itemp);
    }
  }

  std::cout << ">> FEB array --> size: " << febarray.size() << " >> done." << std::endl;
  delete queryFVTH1;
  delete queryFVTH2;
  delete queryFVTH3;
  delete queryFVTH4;
  delete queryFTEMP1;
  session->transaction().commit();
  delete session;
  return febarray;
}

//----------------------------- U X C ------------------------------------------------------------------------
std::vector<RPCObUXC::Item> RPCFw::createUXC(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  std::cout << ">> creating UXC object..." << std::endl;

  // UXCT
  coral::IQuery* queryUXCP = schema.newQuery();
  queryUXCP->addToTableList("RPCGASPARAMETERS");
  queryUXCP->addToTableList("DP_NAME2ID");
  queryUXCP->addToOutputList("DP_NAME2ID.DPNAME", "DPNAME");
  queryUXCP->addToOutputList("DP_NAME2ID.ID", "ID");
  queryUXCP->addToOutputList("RPCGASPARAMETERS.DPID", "DPID");
  queryUXCP->addToOutputList("RPCGASPARAMETERS.CHANGE_DATE", "TSTAMP");
  queryUXCP->addToOutputList("RPCGASPARAMETERS.VALUE", "VALUE");
  coral::IQuery* queryUXCT = schema.newQuery();
  queryUXCT->addToTableList("RPCCOOLING");
  queryUXCT->addToTableList("DP_NAME2ID");
  queryUXCT->addToOutputList("DP_NAME2ID.DPNAME", "DPNAME");
  queryUXCT->addToOutputList("DP_NAME2ID.ID", "ID");
  queryUXCT->addToOutputList("RPCCOOLING.DPID", "DPID");
  queryUXCT->addToOutputList("RPCCOOLING.CHANGE_DATE", "TSTAMP");
  queryUXCT->addToOutputList("RPCCOOLING.VALUE", "VALUE");
  coral::IQuery* queryUXCH = schema.newQuery();
  queryUXCH->addToTableList("RPCCOOLING");
  queryUXCH->addToTableList("DP_NAME2ID");
  queryUXCH->addToOutputList("DP_NAME2ID.DPNAME", "DPNAME");
  queryUXCH->addToOutputList("DP_NAME2ID.ID", "ID");
  queryUXCH->addToOutputList("RPCCOOLING.DPID", "DPID");
  queryUXCH->addToOutputList("RPCCOOLING.CHANGE_DATE", "TSTAMP");
  queryUXCH->addToOutputList("RPCCOOLING.VALUE", "VALUE");

  RPCObUXC::Item Itemp;
  std::vector<RPCObUXC::Item> uxcarray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << "Processing UXC..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionUXCP =
        "RPCGASPARAMETERS.DPID = DP_NAME2ID.ID AND RPCGASPARAMETERS.CHANGE_DATE >:tmin AND "
        "RPCGASPARAMETERS.CHANGE_DATE <:tmax AND (DP_NAME2ID.DPNAME like '%UXCPressure%')";
    queryUXCP->setCondition(conditionUXCP, conditionData);
    coral::ICursor& cursorUXCP = queryUXCP->execute();
    while (cursorUXCP.next()) {
      Itemp.temperature = 0;
      Itemp.pressure = 0;
      Itemp.dewpoint = 0;
      const coral::AttributeList& row = cursorUXCP.currentRow();
      float value = row["VALUE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      unsigned long long ut_time = TtoUT(ts);
      Itemp.pressure = value;
      Itemp.unixtime = ut_time;
      uxcarray.push_back(Itemp);
    }
    std::string conditionUXCT =
        "RPCCOOLING.DPID = DP_NAME2ID.ID AND RPCCOOLING.CHANGE_DATE >:tmin AND RPCCOOLING.CHANGE_DATE <:tmax AND "
        "(DP_NAME2ID.DPNAME like '%TempUXC%')";
    queryUXCT->setCondition(conditionUXCT, conditionData);
    coral::ICursor& cursorUXCT = queryUXCT->execute();
    while (cursorUXCT.next()) {
      Itemp.temperature = 0;
      Itemp.pressure = 0;
      Itemp.dewpoint = 0;
      const coral::AttributeList& row = cursorUXCT.currentRow();
      float value = row["VALUE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      unsigned long long ut_time = TtoUT(ts);
      Itemp.temperature = value;
      Itemp.unixtime = ut_time;
      uxcarray.push_back(Itemp);
    }
    std::string conditionUXCH =
        "RPCCOOLING.DPID = DP_NAME2ID.ID AND RPCCOOLING.CHANGE_DATE >:tmin AND RPCCOOLING.CHANGE_DATE <:tmax AND "
        "(DP_NAME2ID.DPNAME like '%DewpointUXC%')";
    queryUXCH->setCondition(conditionUXCH, conditionData);
    coral::ICursor& cursorUXCH = queryUXCH->execute();
    while (cursorUXCH.next()) {
      Itemp.temperature = 0;
      Itemp.pressure = 0;
      Itemp.dewpoint = 0;
      const coral::AttributeList& row = cursorUXCH.currentRow();
      float value = row["VALUE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      unsigned long long ut_time = TtoUT(ts);
      Itemp.dewpoint = value;
      Itemp.unixtime = ut_time;
      uxcarray.push_back(Itemp);
    }
  } else {
    std::cout << "This objects cannot be copied in this mode. Use RANGE mode!" << std::endl;
  }

  std::cout << ">> UXC array --> size: " << uxcarray.size() << " >> done." << std::endl;
  delete queryUXCT;
  delete queryUXCP;
  delete queryUXCH;
  session->transaction().commit();
  delete session;
  return uxcarray;
}

//----------------------------- M I X ------------------------------------------------------------------------
std::vector<RPCObGasMix::Item> RPCFw::createMix(long long since, long long till) {
  tMIN = UTtoT(since);
  std::cout << ">> Processing since: " << tMIN.day() << "/" << tMIN.month() << "/" << tMIN.year() << " " << tMIN.hour()
            << ":" << tMIN.minute() << "." << tMIN.second() << std::endl;

  coral::ISession* session = this->connect(m_connectionString, m_userName, m_password);
  session->transaction().start(true);
  coral::ISchema& schema = session->nominalSchema();
  std::cout << ">> creating UXC object..." << std::endl;

  coral::IQuery* queryMix = schema.newQuery();
  queryMix->addToTableList("RPCGASPARAMETERS");
  queryMix->addToTableList("DP_NAME2ID");
  queryMix->addToOutputList("DP_NAME2ID.DPNAME", "DPNAME");
  queryMix->addToOutputList("DP_NAME2ID.ID", "ID");
  queryMix->addToOutputList("RPCGASPARAMETERS.DPID", "DPID");
  queryMix->addToOutputList("RPCGASPARAMETERS.CHANGE_DATE", "TSTAMP");
  queryMix->addToOutputList("RPCGASPARAMETERS.VALUE", "VALUE");

  coral::IQuery* queryMix2 = schema.newQuery();
  queryMix2->addToTableList("RPCGASPARAMETERS");
  queryMix2->addToTableList("DP_NAME2ID");
  queryMix2->addToOutputList("DP_NAME2ID.DPNAME", "DPNAME");
  queryMix2->addToOutputList("DP_NAME2ID.ID", "ID");
  queryMix2->addToOutputList("RPCGASPARAMETERS.DPID", "DPID");
  queryMix2->addToOutputList("RPCGASPARAMETERS.CHANGE_DATE", "TSTAMP");
  queryMix2->addToOutputList("RPCGASPARAMETERS.VALUE", "VALUE");

  coral::IQuery* queryMix3 = schema.newQuery();
  queryMix3->addToTableList("RPCGASPARAMETERS");
  queryMix3->addToTableList("DP_NAME2ID");
  queryMix3->addToOutputList("DP_NAME2ID.DPNAME", "DPNAME");
  queryMix3->addToOutputList("DP_NAME2ID.ID", "ID");
  queryMix3->addToOutputList("RPCGASPARAMETERS.DPID", "DPID");
  queryMix3->addToOutputList("RPCGASPARAMETERS.CHANGE_DATE", "TSTAMP");
  queryMix3->addToOutputList("RPCGASPARAMETERS.VALUE", "VALUE");

  RPCObGasMix::Item Mtemp;
  std::vector<RPCObGasMix::Item> marray;
  coral::TimeStamp tlast = tMIN;
  if (till > since) {
    tMAX = UTtoT(till);
    std::cout << ">> Processing till: " << tMAX.day() << "/" << tMAX.month() << "/" << tMAX.year() << " " << tMAX.hour()
              << ":" << tMAX.minute() << "." << tMAX.second() << std::endl;
    std::cout << "Processing UXC..." << std::endl;
    coral::AttributeList conditionData;
    conditionData.extend<coral::TimeStamp>("tmin");
    conditionData.extend<coral::TimeStamp>("tmax");
    conditionData["tmin"].data<coral::TimeStamp>() = tMIN;
    conditionData["tmax"].data<coral::TimeStamp>() = tMAX;
    std::string conditionM1 =
        "RPCGASPARAMETERS.DPID = DP_NAME2ID.ID AND RPCGASPARAMETERS.CHANGE_DATE >:tmin AND "
        "RPCGASPARAMETERS.CHANGE_DATE <:tmax AND (DP_NAME2ID.DPNAME like '%IC4H10Ratio%')";
    queryMix->setCondition(conditionM1, conditionData);
    coral::ICursor& cursorMix = queryMix->execute();
    while (cursorMix.next()) {
      Mtemp.gas1 = 0;
      Mtemp.gas2 = 0;
      Mtemp.gas3 = 0;
      const coral::AttributeList& row = cursorMix.currentRow();
      float value = row["VALUE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      unsigned long long ut_time = TtoUT(ts);
      Mtemp.gas1 = value;
      Mtemp.unixtime = ut_time;
      marray.push_back(Mtemp);
    }
    std::string conditionM2 =
        "RPCGASPARAMETERS.DPID = DP_NAME2ID.ID AND RPCGASPARAMETERS.CHANGE_DATE >:tmin AND "
        "RPCGASPARAMETERS.CHANGE_DATE <:tmax AND (DP_NAME2ID.DPNAME like '%C2H2F4Ratio%')";
    queryMix2->setCondition(conditionM2, conditionData);
    coral::ICursor& cursorMix2 = queryMix2->execute();
    while (cursorMix2.next()) {
      Mtemp.gas1 = 0;
      Mtemp.gas2 = 0;
      Mtemp.gas3 = 0;
      const coral::AttributeList& row = cursorMix2.currentRow();
      float value = row["VALUE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      unsigned long long ut_time = TtoUT(ts);
      Mtemp.gas2 = value;
      Mtemp.unixtime = ut_time;
      marray.push_back(Mtemp);
    }
    std::string conditionM3 =
        "RPCGASPARAMETERS.DPID = DP_NAME2ID.ID AND RPCGASPARAMETERS.CHANGE_DATE >:tmin AND "
        "RPCGASPARAMETERS.CHANGE_DATE <:tmax AND (DP_NAME2ID.DPNAME like '%SF6Ratio%')";
    queryMix3->setCondition(conditionM3, conditionData);
    coral::ICursor& cursorMix3 = queryMix3->execute();
    while (cursorMix3.next()) {
      Mtemp.gas1 = 0;
      Mtemp.gas2 = 0;
      Mtemp.gas3 = 0;
      const coral::AttributeList& row = cursorMix3.currentRow();
      float value = row["VALUE"].data<float>();
      coral::TimeStamp ts = row["TSTAMP"].data<coral::TimeStamp>();
      unsigned long long ut_time = TtoUT(ts);
      Mtemp.gas3 = value;
      Mtemp.unixtime = ut_time;
      marray.push_back(Mtemp);
    }
  } else {
    std::cout << "This objects cannot be copied in this mode. Use RANGE mode!" << std::endl;
  }

  std::cout << ">> GasMix array --> size: " << marray.size() << " >> done." << std::endl;
  delete queryMix;
  session->transaction().commit();
  delete session;
  return marray;
}

//----------------------------------------------------------------------------------------------
coral::TimeStamp RPCFw::UTtoT(long long utime) {
  int yea = static_cast<int>(trunc(utime / 31536000) + 1970);
  int yes = (yea - 1970) * 31536000;
  int cony = ((yea - 1972) % 4) + 1;
  if (cony == 0)
    yes = yes + (yea - 1972) / 4 * 86400;
  else
    yes = yes + static_cast<int>(trunc((yea - 1972) / 4)) * 86400;
  int day = static_cast<int>(trunc((utime - yes) / 86400));
  int rest = static_cast<int>(utime - yes - day * 86400);
  int mon = 0;
  // BISESTILE YEAR
  if (cony == 0) {
    day = day + 1;
    if (day < 32) {
      mon = 1;
      day = day - 0;
    }
    if (day >= 32 && day < 61) {
      mon = 2;
      day = day - 31;
    }
    if (day >= 61 && day < 92) {
      mon = 3;
      day = day - 60;
    }
    if (day >= 92 && day < 122) {
      mon = 4;
      day = day - 91;
    }
    if (day >= 122 && day < 153) {
      mon = 5;
      day = day - 121;
    }
    if (day >= 153 && day < 183) {
      mon = 6;
      day = day - 152;
    }
    if (day >= 183 && day < 214) {
      mon = 7;
      day = day - 182;
    }
    if (day >= 214 && day < 245) {
      mon = 8;
      day = day - 213;
    }
    if (day >= 245 && day < 275) {
      mon = 9;
      day = day - 244;
    }
    if (day >= 275 && day < 306) {
      mon = 10;
      day = day - 274;
    }
    if (day >= 306 && day < 336) {
      mon = 11;
      day = day - 305;
    }
    if (day >= 336) {
      mon = 12;
      day = day - 335;
    }
  }
  // NOT BISESTILE YEAR
  else {
    if (day < 32) {
      mon = 1;
      day = day - 0;
    }
    if (day >= 32 && day < 60) {
      mon = 2;
      day = day - 31;
    }
    if (day >= 60 && day < 91) {
      mon = 3;
      day = day - 59;
    }
    if (day >= 91 && day < 121) {
      mon = 4;
      day = day - 90;
    }
    if (day >= 121 && day < 152) {
      mon = 5;
      day = day - 120;
    }
    if (day >= 152 && day < 182) {
      mon = 6;
      day = day - 151;
    }
    if (day >= 182 && day < 213) {
      mon = 7;
      day = day - 181;
    }
    if (day >= 213 && day < 244) {
      mon = 8;
      day = day - 212;
    }
    if (day >= 244 && day < 274) {
      mon = 9;
      day = day - 243;
    }
    if (day >= 274 && day < 305) {
      mon = 10;
      day = day - 273;
    }
    if (day >= 305 && day < 335) {
      mon = 11;
      day = day - 304;
    }
    if (day >= 335) {
      mon = 12;
      day = day - 334;
    }
  }

  int hou = static_cast<int>(trunc(rest / 3600));
  rest = rest - hou * 3600;
  int min = static_cast<int>(trunc(rest / 60));
  rest = rest - min * 60;
  int sec = rest;
  int nan = 0;

  //  std::cout <<">> Processing since: "<<day<<"/"<<mon<<"/"<<yea<<" "<<hou<<":"<<min<<"."<<sec<< std::endl;

  coral::TimeStamp Tthr;

  Tthr = coral::TimeStamp(yea, mon, day, hou, min, sec, nan);
  return Tthr;
}

//----------------------------------------------------------------------------------------------
unsigned long long RPCFw::TtoUT(const coral::TimeStamp& time) {
  long long utime = (time.year() - 1970) * 31536000 + static_cast<int>(trunc((time.year() - 1972) / 4)) * 86400 +
                    (((time.month() - 1) * 31) * 86400) + ((time.day() - 1) * 86400) + time.hour() * 3600 +
                    time.minute() * 60 + time.second();

  if (time.month() == 3)
    utime = utime - 3 * 86400;
  if (time.month() == 4)
    utime = utime - 3 * 86400;
  if (time.month() == 5)
    utime = utime - 4 * 86400;
  if (time.month() == 6)
    utime = utime - 4 * 86400;
  if (time.month() == 7)
    utime = utime - 5 * 86400;
  if (time.month() == 8)
    utime = utime - 5 * 86400;
  if (time.month() == 9)
    utime = utime - 5 * 86400;
  if (time.month() == 10)
    utime = utime - 6 * 86400;
  if (time.month() == 11)
    utime = utime - 6 * 86400;
  if (time.month() == 12)
    utime = utime - 7 * 86400;

  return utime;
}

bool RPCFw::isMajor(const coral::TimeStamp& fir, const coral::TimeStamp& sec) {
  double first =
      fir.year() * 1e10 + fir.month() + 1e8 + fir.day() * 1e6 + fir.hour() * 1e4 + fir.minute() * 1e2 + fir.second();
  double second =
      sec.year() * 1e10 + sec.month() + 1e8 + sec.day() * 1e6 + sec.hour() * 1e4 + sec.minute() * 1e2 + sec.second();

  if (first > second)
    return true;

  return false;
}

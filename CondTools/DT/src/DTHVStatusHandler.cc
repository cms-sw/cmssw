/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/01/27 15:36:19 $
 *  $Revision: 1.12 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTHVStatusHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondTools/DT/interface/DTHVAbstractCheck.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/TimeStamp.h"

//---------------
// C++ Headers --
//---------------
#include <map>
#include <sys/time.h>

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
DTHVStatusHandler::DTHVStatusHandler(const edm::ParameterSet& ps)
    : dataTag(ps.getParameter<std::string>("tag")),
      onlineConnect(ps.getParameter<std::string>("onlineDB")),
      utilConnect(ps.getParameter<std::string>("utilDB")),
      onlineAuthentication(ps.getParameter<std::string>("onlineAuthentication")),
      bufferConnect(ps.getParameter<std::string>("bufferDB")),
      ySince(ps.getParameter<int>("sinceYear")),
      mSince(ps.getParameter<int>("sinceMonth")),
      dSince(ps.getParameter<int>("sinceDay")),
      hSince(ps.getParameter<int>("sinceHour")),
      pSince(ps.getParameter<int>("sinceMinute")),
      sSince(ps.getParameter<int>("sinceSecond")),
      yUntil(ps.getParameter<int>("untilYear")),
      mUntil(ps.getParameter<int>("untilMonth")),
      dUntil(ps.getParameter<int>("untilDay")),
      hUntil(ps.getParameter<int>("untilHour")),
      pUntil(ps.getParameter<int>("untilMinute")),
      sUntil(ps.getParameter<int>("untilSecond")),
      dumpAtStart(ps.getParameter<bool>("dumpAtStart")),
      dumpAtEnd(ps.getParameter<bool>("dumpAtEnd")),
      bwdTime(ps.getParameter<long long int>("bwdTime")),
      fwdTime(ps.getParameter<long long int>("fwdTime")),
      minTime(ps.getParameter<long long int>("minTime")),
      omds_session(),
      util_session(),
      buff_session(),
      mapVersion(ps.getParameter<std::string>("mapVersion")),
      splitVersion(ps.getParameter<std::string>("splitVersion")) {
  std::cout << " PopCon application for DT HV data export " << onlineAuthentication << std::endl;
  hvChecker = DTHVAbstractCheck::getInstance();
  maxPayload = 1000;
}

//--------------
// Destructor --
//--------------
DTHVStatusHandler::~DTHVStatusHandler() {}

//--------------
// Operations --
//--------------
void DTHVStatusHandler::getNewObjects() {
  std::cout << "get new objects..." << std::endl;

  // online DB connection - data
  std::cout << "configure omds DbConnection" << std::endl;
  cond::persistency::ConnectionPool connection;
  //  conn->configure( cond::CmsDefaults );
  connection.setAuthenticationPath(onlineAuthentication);
  connection.configure();
  std::cout << "create omds DbSession" << std::endl;
  omds_session = connection.createSession(onlineConnect);
  std::cout << "start omds transaction" << std::endl;
  omds_session.transaction().start();
  std::cout << "" << std::endl;

  // online DB connection - util
  std::cout << "create util DbSession" << std::endl;
  util_session = connection.createSession(onlineConnect);
  std::cout << "startutil  transaction" << std::endl;
  util_session.transaction().start();
  std::cout << "" << std::endl;

  // buffer DB connection
  std::cout << "create buffer DbSession" << std::endl;
  buff_session = connection.createSession(bufferConnect);
  std::cout << "start buffer transaction" << std::endl;
  buff_session.transaction().start();

  // offline info

  //to access the information on the tag inside the offline database:
  cond::TagInfo const& ti = tagInfo();
  cond::Time_t last = ti.lastInterval.first;
  std::cout << "latest DCS data (HV) already copied until: " << last << std::endl;

  coral::TimeStamp coralSince(ySince, mSince, dSince, hSince, pSince, sSince, 0);
  procSince = condTime(coralSince);
  coral::TimeStamp coralUntil(yUntil, mUntil, dUntil, hUntil, pUntil, sUntil, 0);
  procUntil = condTime(coralUntil);
  lastFound = 0;
  nextFound = 0;
  timeLimit = 0;
  lastStamp = 0;

  if (last == 0) {
    DTHVStatus* dummyStatus = new DTHVStatus(dataTag);
    cond::Time_t snc = 1;
    m_to_transfer.push_back(std::make_pair(dummyStatus, snc));
    last = procSince + 1;
    std::cout << "no old data... " << last << std::endl;
  } else {
    Ref payload = lastPayload();
    DTHVStatus::const_iterator paylIter = payload->begin();
    DTHVStatus::const_iterator paylIend = payload->end();
    while (paylIter != paylIend) {
      const std::pair<DTHVStatusId, DTHVStatusData>& entry = *paylIter++;
      const DTHVStatusId& chan = entry.first;
      const DTHVStatusData& data = entry.second;
      DTWireId id(chan.wheelId, chan.stationId, chan.sectorId, chan.slId, chan.layerId, chan.partId + 10);
      hvChecker->setStatus(id.rawId(), data.flagA, data.flagC, data.flagS, snapshotValues, aliasMap, layerMap);
    }
  }
  coral::TimeStamp coralLast = coralTime(last);
  coral::TimeStamp coralProc = coral::TimeStamp::now();
  cond::Time_t condProc = condTime(coralProc);

  if (procSince > condProc) {
    std::cout << "Required time interval in the future: " << std::endl
              << " copy since " << ySince << " " << mSince << " " << dSince << " ( " << procSince << " )" << std::endl
              << " current time " << coralProc.year() << " " << coralProc.month() << " " << coralProc.day()
              << std::endl;
  }
  if (procUntil > condProc)
    procUntil = condProc;
  if (last > procSince) {
    if (last < procUntil) {
      procSince = last;
      checkNewData();
    } else {
      std::cout << "Required time interval already copied: " << std::endl
                << " copy until " << yUntil << " " << mUntil << " " << dUntil << " ( " << procUntil << " )" << std::endl
                << " data until " << coralLast.year() << " " << coralLast.month() << " " << coralLast.day()
                << std::endl;
    }
  } else {
    std::cout << "Required time interval not contiguous with copied data: " << std::endl
              << " data until " << coralLast.year() << " " << coralLast.month() << " " << coralLast.day() << std::endl
              << " copy since " << ySince << " " << mSince << " " << dSince << " ( " << procSince << " )" << std::endl;
  }

  buff_session.transaction().commit();
  buff_session.close();
  omds_session.close();
  util_session.close();

  return;
}

void DTHVStatusHandler::checkNewData() {
  //to access the information on last successful log entry for this tag:
  //  cond::LogDBEntry const & lde = logDBEntry();

  //to access the lastest payload (Ref is a smart pointer)
  //  Ref payload = lastPayload();

  std::cout << "check for new data since " << procSince << " " << coralTime(procSince).total_nanoseconds() << " "
            << coralTime(procSince).year() << " " << coralTime(procSince).month() << " " << coralTime(procSince).day()
            << " " << coralTime(procSince).hour() << " " << coralTime(procSince).minute() << " "
            << coralTime(procSince).second() << std::endl;
  std::cout << "                   until " << procUntil << " " << coralTime(procUntil).total_nanoseconds() << " "
            << coralTime(procUntil).year() << " " << coralTime(procUntil).month() << " " << coralTime(procUntil).day()
            << " " << coralTime(procUntil).hour() << " " << coralTime(procUntil).minute() << " "
            << coralTime(procUntil).second() << std::endl;

  std::set<std::string> omds_lt(omds_session.nominalSchema().listTables());
  std::set<std::string>::const_iterator omds_iter = omds_lt.begin();
  std::set<std::string>::const_iterator omds_iend = omds_lt.end();
  while (omds_iter != omds_iend) {
    const std::string& istr = *omds_iter++;
    std::cout << "TABLE: " << istr << std::endl;
  }

  std::set<std::string> util_lt(util_session.nominalSchema().listTables());
  std::set<std::string>::const_iterator util_iter = util_lt.begin();
  std::set<std::string>::const_iterator util_iend = util_lt.end();
  while (util_iter != util_iend) {
    const std::string& istr = *util_iter++;
    std::cout << "TABLE: " << istr << std::endl;
  }

  getLayerSplit();
  getChannelMap();
  getChannelSplit();

  std::cout << "open buffer db..." << std::endl;

  if (!(buff_session.nominalSchema().existsTable("HVSNAPSHOT")))
    createSnapshot();
  updateHVStatus();

  return;
}

std::string DTHVStatusHandler::id() const { return "DTHVStatusHandler"; }

void DTHVStatusHandler::getChannelMap() {
  if (!(buff_session.nominalSchema().existsTable("HVALIASES"))) {
    dumpHVAliases();
  } else {
    std::cout << "retrieve aliases table..." << std::endl;
    coral::ITable& hvalTable = buff_session.nominalSchema().tableHandle("HVALIASES");
    std::unique_ptr<coral::IQuery> hvalQuery(hvalTable.newQuery());
    hvalQuery->addToOutputList("DETID");
    hvalQuery->addToOutputList("DPID");
    coral::ICursor& hvalCursor = hvalQuery->execute();
    int chId;
    int dpId;
    while (hvalCursor.next()) {
      chId = hvalCursor.currentRow()["DETID"].data<int>();
      dpId = hvalCursor.currentRow()["DPID"].data<int>();
      aliasMap.insert(std::pair<int, int>(dpId, chId));
      layerMap.insert(std::pair<int, int>(chId, dpId));
    }
  }

  return;
}

void DTHVStatusHandler::getLayerSplit() {
  std::cout << "retrieve layer split table..." << std::endl;
  int whe;
  int sec;
  int sta;
  int qua;
  int lay;
  int l_p;
  int f_c;
  int l_c;
  coral::ITable& lsplTable = util_session.nominalSchema().tableHandle("DT_HV_LAYER_SPLIT");
  std::cout << "         layer split table got..." << std::endl;
  std::unique_ptr<coral::IQuery> lsplQuery(lsplTable.newQuery());
  coral::AttributeList versionBindVariableList;
  versionBindVariableList.extend("version", typeid(std::string));
  versionBindVariableList["version"].data<std::string>() = mapVersion;
  lsplQuery->setCondition("VERSION=:version", versionBindVariableList);
  lsplQuery->addToOutputList("WHEEL");
  lsplQuery->addToOutputList("SECTOR");
  lsplQuery->addToOutputList("STATION");
  lsplQuery->addToOutputList("SUPERLAYER");
  lsplQuery->addToOutputList("LAYER");
  lsplQuery->addToOutputList("PART");
  lsplQuery->addToOutputList("FIRST_CELL");
  lsplQuery->addToOutputList("LAST_CELL");
  coral::ICursor& lsplCursor = lsplQuery->execute();
  while (lsplCursor.next()) {
    whe = lsplCursor.currentRow()["WHEEL"].data<int>();
    sec = lsplCursor.currentRow()["SECTOR"].data<int>();
    sta = lsplCursor.currentRow()["STATION"].data<int>();
    qua = lsplCursor.currentRow()["SUPERLAYER"].data<int>();
    lay = lsplCursor.currentRow()["LAYER"].data<int>();
    l_p = lsplCursor.currentRow()["PART"].data<int>();
    f_c = lsplCursor.currentRow()["FIRST_CELL"].data<int>();
    l_c = lsplCursor.currentRow()["LAST_CELL"].data<int>();
    DTWireId wireId(whe, sta, sec, qua, lay, 10 + l_p);
    laySplit.insert(std::pair<int, int>(wireId.rawId(), (f_c * 10000) + l_c));
  }
  std::cout << "channel split table retrieved" << std::endl;
  return;
}

void DTHVStatusHandler::getChannelSplit() {
  std::cout << "retrieve channel split table..." << std::endl;
  int pwhe;
  int psec;
  int psta;
  int pqua;
  int play;
  int pl_p;
  int swhe;
  int ssec;
  int ssta;
  int squa;
  int slay;
  int sl_p;
  coral::ITable& csplTable = util_session.nominalSchema().tableHandle("DT_HV_CHANNEL_SPLIT");
  std::unique_ptr<coral::IQuery> csplQuery(csplTable.newQuery());
  coral::AttributeList versionBindVariableList;
  versionBindVariableList.extend("version", typeid(std::string));
  versionBindVariableList["version"].data<std::string>() = splitVersion;
  csplQuery->setCondition("VERSION=:version", versionBindVariableList);
  csplQuery->addToOutputList("P_WHEEL");
  csplQuery->addToOutputList("P_SECTOR");
  csplQuery->addToOutputList("P_STATION");
  csplQuery->addToOutputList("P_SUPERLAYER");
  csplQuery->addToOutputList("P_LAYER");
  csplQuery->addToOutputList("P_PART");
  csplQuery->addToOutputList("S_NUMBER");
  csplQuery->addToOutputList("S_WHEEL");
  csplQuery->addToOutputList("S_SECTOR");
  csplQuery->addToOutputList("S_STATION");
  csplQuery->addToOutputList("S_SUPERLAYER");
  csplQuery->addToOutputList("S_LAYER");
  csplQuery->addToOutputList("S_PART");
  coral::ICursor& csplCursor = csplQuery->execute();
  while (csplCursor.next()) {
    pwhe = csplCursor.currentRow()["P_WHEEL"].data<int>();
    psec = csplCursor.currentRow()["P_SECTOR"].data<int>();
    psta = csplCursor.currentRow()["P_STATION"].data<int>();
    pqua = csplCursor.currentRow()["P_SUPERLAYER"].data<int>();
    play = csplCursor.currentRow()["P_LAYER"].data<int>();
    pl_p = csplCursor.currentRow()["P_PART"].data<int>();
    csplCursor.currentRow()["S_NUMBER"].data<int>();
    swhe = csplCursor.currentRow()["S_WHEEL"].data<int>();
    ssec = csplCursor.currentRow()["S_SECTOR"].data<int>();
    ssta = csplCursor.currentRow()["S_STATION"].data<int>();
    squa = csplCursor.currentRow()["S_SUPERLAYER"].data<int>();
    slay = csplCursor.currentRow()["S_LAYER"].data<int>();
    sl_p = csplCursor.currentRow()["S_PART"].data<int>();
    DTWireId pId(pwhe, psta, psec, pqua, play, 10 + pl_p);
    DTWireId sId(swhe, ssta, ssec, squa, slay, 10 + sl_p);
    int pRaw = pId.rawId();
    int sRaw = sId.rawId();
    std::vector<int>* splitList = nullptr;
    std::map<int, std::vector<int>*>::iterator iter = channelSplit.find(pRaw);
    std::map<int, std::vector<int>*>::iterator iend = channelSplit.end();
    if (iter == iend) {
      channelSplit.insert(std::pair<int, std::vector<int>*>(pRaw, splitList = new std::vector<int>));
    } else {
      splitList = iter->second;
    }
    splitList->push_back(sRaw);
  }
  return;
}

void DTHVStatusHandler::dumpHVAliases() {
  std::cout << "DTHVStatusHandler::dumpHVAliases - begin" << std::endl;

  std::cout << "create aliases description..." << std::endl;
  coral::TableDescription hvalDesc;
  hvalDesc.setName("HVALIASES");
  hvalDesc.insertColumn("DETID", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvalDesc.insertColumn("DPID", coral::AttributeSpecification::typeNameForId(typeid(int)));
  std::cout << "create aliases table..." << std::endl;
  coral::ITable& hvalTable = buff_session.nominalSchema().createTable(hvalDesc);

  std::cout << "open DPNAME table..." << std::endl;
  std::map<int, std::string> idMap;
  coral::ITable& dpidTable = omds_session.nominalSchema().tableHandle("DP_NAME2ID");
  std::unique_ptr<coral::IQuery> dpidQuery(dpidTable.newQuery());
  dpidQuery->addToOutputList("ID");
  dpidQuery->addToOutputList("DPNAME");
  coral::ICursor& dpidCursor = dpidQuery->execute();
  while (dpidCursor.next()) {
    const coral::AttributeList& row = dpidCursor.currentRow();
    int id = static_cast<int>(0.01 + row["ID"].data<float>());
    std::string dp = row["DPNAME"].data<std::string>();
    idMap.insert(std::pair<int, std::string>(id, dp));
  }
  std::cout << "DPNAME table read... " << idMap.size() << std::endl;

  std::cout << "open ALIASES table..." << std::endl;
  std::map<std::string, std::string> cnMap;
  coral::ITable& nameTable = omds_session.nominalSchema().tableHandle("ALIASES");
  std::unique_ptr<coral::IQuery> nameQuery(nameTable.newQuery());
  nameQuery->addToOutputList("DPE_NAME");
  nameQuery->addToOutputList("ALIAS");
  coral::ICursor& nameCursor = nameQuery->execute();
  while (nameCursor.next()) {
    const coral::AttributeList& row = nameCursor.currentRow();
    std::string dp = row["DPE_NAME"].data<std::string>();
    std::string an = row["ALIAS"].data<std::string>();
    if (an.length() < 20)
      continue;
    cnMap.insert(std::pair<std::string, std::string>(dp, an));
  }
  std::cout << "ALIASES table read... " << cnMap.size() << std::endl;

  std::map<int, std::string>::const_iterator idIter = idMap.begin();
  std::map<int, std::string>::const_iterator idIend = idMap.end();
  std::string outChk("/outputChannel");
  while (idIter != idIend) {
    const std::pair<int, std::string>& ientry = *idIter++;
    int dpId = ientry.first;
    std::string dp = ientry.second;
    int ldp = dp.length();
    if (ldp < 20)
      continue;
    std::string subOut(dp.substr(ldp - 17, 17));
    std::string subChk(subOut.substr(0, 14));
    if (subChk != outChk)
      continue;
    std::string chName(dp.substr(0, ldp - 17));
    chName += ".actual.OvC";
    int chCode = subOut.c_str()[16] - '0';
    std::map<std::string, std::string>::const_iterator jter = cnMap.find(chName);
    if (jter == cnMap.end())
      continue;
    const std::pair<std::string, std::string>& jentry = *jter;
    std::cout << dp << std::endl << chName << " " << chCode << std::endl;
    std::string an(jentry.second);
    int al = an.length();
    int iofw = 7 + an.find("DT_HV_W", 0);
    int iofc = 3 + an.find("_MB", 0);
    int iofs = 2 + an.find("_S", 0);
    int iofq = 3 + an.find("_SL", 0);
    int iofl = 2 + an.find("_L", 0);
    if ((iofw == al) || (iofc == al) || (iofs == al) || (iofq == al) || (iofl == al)) {
      break;
    }
    int ioew = an.find("_", iofw);
    int ioec = an.find("_", iofc);
    int ioes = an.find("_", iofs);
    int ioeq = an.find("_", iofq);
    int ioel = an.find("_", iofl);
    std::string swhe(an.substr(iofw, ioew - iofw));
    const char* cwhe = swhe.c_str();
    int whe = cwhe[1] - '0';
    if (*cwhe != 'P')
      whe = -whe;

    std::string scha(an.substr(iofc, ioec - iofc));
    const char* ccha = scha.c_str();
    int cha = *ccha - '0';

    std::string ssec(an.substr(iofs, ioes - iofs));
    const char* csec = ssec.c_str();
    int sec = ((*csec - '0') * 10) + (csec[1] - '0');
    if ((csec[2] == 'R') && (sec == 10))
      sec = 14;
    if ((csec[2] == 'L') && (sec == 4))
      sec = 13;

    std::string squa(an.substr(iofq, ioeq - iofq));
    const char* cqua = squa.c_str();
    int qua = *cqua - '0';

    std::string slay(an.substr(iofl, ioel - iofl));
    const char* clay = slay.c_str();
    int lay = *clay - '0';

    DTWireId wireId(whe, cha, sec, qua, lay, 10 + chCode);
    int chId = wireId.rawId();
    coral::AttributeList newChan;
    newChan.extend("DETID", typeid(int));
    newChan.extend("DPID", typeid(int));
    newChan["DETID"].data<int>() = chId;
    newChan["DPID"].data<int>() = dpId;
    hvalTable.dataEditor().insertRow(newChan);
    aliasMap.insert(std::pair<int, int>(dpId, chId));
    layerMap.insert(std::pair<int, int>(chId, dpId));
  }

  std::cout << "DTHVStatusHandler::dumpHVAliases - end" << std::endl;
  return;
}

void DTHVStatusHandler::createSnapshot() {
  std::cout << "create snapshot description..." << std::endl;
  coral::TableDescription hvssDesc;
  hvssDesc.setName("HVSNAPSHOT");
  hvssDesc.insertColumn("TIME", coral::AttributeSpecification::typeNameForId(typeid(coral::TimeStamp)));
  hvssDesc.insertColumn("WHEEL", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("STATION", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("SECTOR", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("SUPERLAYER", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("LAYER", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("CHAN", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("TYPE", coral::AttributeSpecification::typeNameForId(typeid(int)));
  hvssDesc.insertColumn("VALUE", coral::AttributeSpecification::typeNameForId(typeid(float)));
  std::cout << "create snapshot table..." << std::endl;
  buff_session.nominalSchema().createTable(hvssDesc);
  coral::ITable& bufferTable = buff_session.nominalSchema().tableHandle("HVSNAPSHOT");
  coral::AttributeList newMeas;
  newMeas.extend("TIME", typeid(coral::TimeStamp));
  newMeas.extend("WHEEL", typeid(int));
  newMeas.extend("STATION", typeid(int));
  newMeas.extend("SECTOR", typeid(int));
  newMeas.extend("SUPERLAYER", typeid(int));
  newMeas.extend("LAYER", typeid(int));
  newMeas.extend("CHAN", typeid(int));
  newMeas.extend("TYPE", typeid(int));
  newMeas.extend("VALUE", typeid(float));

  long long int zeroTime = 0LL;
  newMeas["TIME"].data<coral::TimeStamp>() = coral::TimeStamp(zeroTime);
  newMeas["VALUE"].data<float>() = -999999.0;

  std::map<int, int>::const_iterator iter = aliasMap.begin();
  std::map<int, int>::const_iterator iend = aliasMap.end();
  while (iter != iend) {
    const std::pair<int, int>& entry = *iter++;
    int detId = entry.second;
    DTWireId chlId(detId);
    newMeas["WHEEL"].data<int>() = chlId.wheel();
    newMeas["STATION"].data<int>() = chlId.station();
    newMeas["SECTOR"].data<int>() = chlId.sector();
    newMeas["SUPERLAYER"].data<int>() = chlId.superLayer();
    newMeas["LAYER"].data<int>() = chlId.layer();
    newMeas["CHAN"].data<int>() = chlId.wire() - 10;
    int itype;
    for (itype = 1; itype <= 2; itype++) {
      newMeas["TYPE"].data<int>() = itype;
      bufferTable.dataEditor().insertRow(newMeas);
    }
  }

  std::cout << "create logging info..." << std::endl;
  if (buff_session.nominalSchema().existsTable("LOG"))
    buff_session.nominalSchema().dropTable("LOG");
  coral::TableDescription infoDesc;
  infoDesc.setName("LOG");
  infoDesc.insertColumn("EXECTIME", coral::AttributeSpecification::typeNameForId(typeid(coral::TimeStamp)));
  infoDesc.insertColumn("SNAPSHOT", coral::AttributeSpecification::typeNameForId(typeid(coral::TimeStamp)));
  buff_session.nominalSchema().createTable(infoDesc);
  coral::AttributeList newInfo;
  newInfo.extend("EXECTIME", typeid(coral::TimeStamp));
  newInfo.extend("SNAPSHOT", typeid(coral::TimeStamp));
  newInfo["EXECTIME"].data<coral::TimeStamp>() = coral::TimeStamp(zeroTime);
  newInfo["SNAPSHOT"].data<coral::TimeStamp>() = coral::TimeStamp(zeroTime);
  coral::ITable& infoTable = buff_session.nominalSchema().tableHandle("LOG");
  infoTable.dataEditor().insertRow(newInfo);

  return;
}

void DTHVStatusHandler::updateHVStatus() {
  int missingChannels = recoverSnapshot();
  cond::Time_t snapshotTime = recoverLastTime();
  std::cout << " snapshot at " << snapshotTime << " ( " << coralTime(snapshotTime).total_nanoseconds() << " ) "
            << std::endl;
  if (snapshotTime > procSince) {
    coral::TimeStamp coralSnap = coralTime(snapshotTime);
    std::cout << "too recent snapshot: " << std::endl
              << " snapshot at " << coralSnap.year() << " " << coralSnap.month() << " " << coralSnap.day() << std::endl
              << " copy since " << ySince << " " << mSince << " " << dSince << " ( " << procSince << " )" << std::endl;
    return;
  }
  long long int dTime = bwdTime;
  dTime <<= 32;
  cond::Time_t condUntil = procSince;
  cond::Time_t condSince = condUntil - dTime;

  while (missingChannels) {
    std::cout << "back iteration: " << condSince << " ( " << coralTime(condSince).total_nanoseconds() << " ) -> "
              << condUntil << " ( " << coralTime(condUntil).total_nanoseconds() << " ) " << std::endl;
    if (condSince <= snapshotTime)
      condSince = snapshotTime;
    std::cout << "corrected since: " << condSince << " ( " << coralTime(condSince).total_nanoseconds() << " ) "
              << std::endl;
    if (condSince >= condUntil)
      break;
    std::cout << "missing... " << missingChannels << std::endl;
    checkForPeriod(condSince, condUntil, missingChannels, false);
    condUntil = condSince;
    condSince = condUntil - dTime;
  }

  if (dumpAtStart)
    dumpSnapshot(coralTime(procSince));

  copyHVData();

  if (dumpAtEnd)
    dumpSnapshot(coralTime(lastFound));

  return;
}

int DTHVStatusHandler::recoverSnapshot() {
  int missingChannels = 0;
  std::map<int, int>::const_iterator layIter = layerMap.begin();
  std::map<int, int>::const_iterator layIend = layerMap.end();
  std::cout << "retrieve snapshot table..." << std::endl;
  coral::ITable& hvssTable = buff_session.nominalSchema().tableHandle("HVSNAPSHOT");
  std::unique_ptr<coral::IQuery> hvssQuery(hvssTable.newQuery());
  hvssQuery->addToOutputList("TIME");
  hvssQuery->addToOutputList("WHEEL");
  hvssQuery->addToOutputList("STATION");
  hvssQuery->addToOutputList("SECTOR");
  hvssQuery->addToOutputList("SUPERLAYER");
  hvssQuery->addToOutputList("LAYER");
  hvssQuery->addToOutputList("CHAN");
  hvssQuery->addToOutputList("TYPE");
  hvssQuery->addToOutputList("VALUE");
  coral::ICursor& hvssCursor = hvssQuery->execute();
  while (hvssCursor.next()) {
    coral::TimeStamp time = hvssCursor.currentRow()["TIME"].data<coral::TimeStamp>();
    int whe = hvssCursor.currentRow()["WHEEL"].data<int>();
    int sta = hvssCursor.currentRow()["STATION"].data<int>();
    int sec = hvssCursor.currentRow()["SECTOR"].data<int>();
    int qua = hvssCursor.currentRow()["SUPERLAYER"].data<int>();
    int lay = hvssCursor.currentRow()["LAYER"].data<int>();
    int l_p = hvssCursor.currentRow()["CHAN"].data<int>();
    int mty = hvssCursor.currentRow()["TYPE"].data<int>();
    float value = hvssCursor.currentRow()["VALUE"].data<float>();
    if (mty > 2)
      continue;
    DTWireId wireId(whe, sta, sec, qua, lay, 10 + l_p);
    layIter = layerMap.find(wireId.rawId());
    if (layIter == layIend) {
      continue;
    }
    int dpId = (layIter->second * 10) + mty;
    snapshotValues.insert(std::pair<int, timedMeasurement>(dpId, timedMeasurement(time.total_nanoseconds(), value)));
    missingChannels++;
  }
  return missingChannels;
}

cond::Time_t DTHVStatusHandler::recoverLastTime() {
  long long int lastTime = 0LL;
  long long int chanTime = 0LL;
  std::map<int, timedMeasurement>::iterator mapIter = snapshotValues.begin();
  std::map<int, timedMeasurement>::iterator mapIend = snapshotValues.end();
  while (mapIter != mapIend) {
    const std::pair<int, timedMeasurement>& entry = *mapIter++;
    chanTime = entry.second.first;
    if (lastTime < chanTime)
      lastTime = chanTime;
  }
  return condTime(lastTime);
  /*
  coral::ITable& infoTable =
         buff_session.nominalSchema().tableHandle( "LOG" );
  std::unique_ptr<coral::IQuery> infoQuery( infoTable.newQuery() );
  infoQuery->addToOutputList( "SNAPSHOT" );
  coral::ICursor& infoCursor = infoQuery->execute();
  coral::TimeStamp time;
  while ( infoCursor.next() ) {
    time = infoCursor.currentRow()["SNAPSHOT"].data<coral::TimeStamp>();
  }
  return condTime( time );
*/
}

void DTHVStatusHandler::dumpSnapshot(const coral::TimeStamp& time) {
  std::cout << "dump snapshot to buffer db..." << std::endl;
  std::string emptyCondition("");
  coral::AttributeList emptyBindVariableList;
  std::map<int, int>::const_iterator mapIter = aliasMap.begin();
  std::map<int, int>::const_iterator mapIend = aliasMap.end();
  coral::ITable& hvssTable = buff_session.nominalSchema().tableHandle("HVSNAPSHOT");
  coral::ITableDataEditor& hvssEditor(hvssTable.dataEditor());
  long nRows = hvssEditor.deleteRows(emptyCondition, emptyBindVariableList);
  std::cout << nRows << " rows deleted" << std::endl;

  coral::AttributeList newMeas;
  newMeas.extend("TIME", typeid(coral::TimeStamp));
  newMeas.extend("WHEEL", typeid(int));
  newMeas.extend("STATION", typeid(int));
  newMeas.extend("SECTOR", typeid(int));
  newMeas.extend("SUPERLAYER", typeid(int));
  newMeas.extend("LAYER", typeid(int));
  newMeas.extend("CHAN", typeid(int));
  newMeas.extend("TYPE", typeid(int));
  newMeas.extend("VALUE", typeid(float));

  nRows = 0;
  std::map<int, timedMeasurement>::const_iterator ssvIter = snapshotValues.begin();
  std::map<int, timedMeasurement>::const_iterator ssvIend = snapshotValues.end();
  while (ssvIter != ssvIend) {
    const std::pair<int, timedMeasurement>& entry = *ssvIter++;
    int dpty = entry.first;
    int dpId = dpty / 10;
    int type = dpty % 10;
    mapIter = aliasMap.find(dpId);
    if (mapIter == mapIend)
      continue;
    DTWireId chlId(mapIter->second);
    const timedMeasurement& tMeas = entry.second;
    long long int newTime = tMeas.first;
    newMeas["TIME"].data<coral::TimeStamp>() = coral::TimeStamp(newTime);
    newMeas["WHEEL"].data<int>() = chlId.wheel();
    newMeas["STATION"].data<int>() = chlId.station();
    newMeas["SECTOR"].data<int>() = chlId.sector();
    newMeas["SUPERLAYER"].data<int>() = chlId.superLayer();
    newMeas["LAYER"].data<int>() = chlId.layer();
    newMeas["CHAN"].data<int>() = chlId.wire() - 10;
    newMeas["TYPE"].data<int>() = type;
    newMeas["VALUE"].data<float>() = tMeas.second;
    hvssEditor.insertRow(newMeas);
    nRows++;
  }
  std::cout << nRows << " rows updated" << std::endl;

  std::cout << "create logging info..." << std::endl;
  if (buff_session.nominalSchema().existsTable("LOG"))
    buff_session.nominalSchema().dropTable("LOG");
  coral::TableDescription infoDesc;
  infoDesc.setName("LOG");
  infoDesc.insertColumn("EXECTIME", coral::AttributeSpecification::typeNameForId(typeid(coral::TimeStamp)));
  infoDesc.insertColumn("SNAPSHOT", coral::AttributeSpecification::typeNameForId(typeid(coral::TimeStamp)));
  buff_session.nominalSchema().createTable(infoDesc);
  coral::AttributeList newInfo;
  newInfo.extend("EXECTIME", typeid(coral::TimeStamp));
  newInfo.extend("SNAPSHOT", typeid(coral::TimeStamp));
  newInfo["EXECTIME"].data<coral::TimeStamp>() = coral::TimeStamp::now();
  newInfo["SNAPSHOT"].data<coral::TimeStamp>() = time;
  coral::ITable& infoTable = buff_session.nominalSchema().tableHandle("LOG");
  infoTable.dataEditor().insertRow(newInfo);

  return;
}

int DTHVStatusHandler::checkForPeriod(cond::Time_t condSince,
                                      cond::Time_t condUntil,
                                      int& missingChannels,
                                      bool copyOffline) {
  std::map<int, timedMeasurement>::iterator mapIter = snapshotValues.begin();
  std::map<int, timedMeasurement>::iterator mapIend = snapshotValues.end();

  std::map<long long int, channelValue> periodBuffer;

  coral::ITable& fwccTable = omds_session.nominalSchema().tableHandle("FWCAENCHANNEL");
  std::unique_ptr<coral::IQuery> fwccQuery(fwccTable.newQuery());
  fwccQuery->addToOutputList("DPID");
  fwccQuery->addToOutputList("CHANGE_DATE");
  fwccQuery->addToOutputList("ACTUAL_VMON");
  fwccQuery->addToOutputList("ACTUAL_IMON");
  fwccQuery->addToOutputList("ACTUAL_ISON");
  fwccQuery->addToOutputList("ACTUAL_STATUS");
  fwccQuery->addToOutputList("ACTUAL_OVC");
  coral::AttributeList timeBindVariableList;
  timeBindVariableList.extend("since", typeid(coral::TimeStamp));
  timeBindVariableList.extend("until", typeid(coral::TimeStamp));
  coral::TimeStamp coralSince = coralTime(condSince);
  coral::TimeStamp coralUntil = coralTime(condUntil);
  std::cout << "look for data since " << coralSince.year() << " " << coralSince.month() << " " << coralSince.day()
            << " " << coralSince.hour() << ":" << coralSince.minute() << ":" << coralSince.second() << " until "
            << coralUntil.year() << " " << coralUntil.month() << " " << coralUntil.day() << " " << coralUntil.hour()
            << ":" << coralUntil.minute() << ":" << coralUntil.second() << std::endl;
  timeBindVariableList["since"].data<coral::TimeStamp>() = coralTime(condSince);
  timeBindVariableList["until"].data<coral::TimeStamp>() = coralTime(condUntil);
  fwccQuery->setCondition("CHANGE_DATE>:since and CHANGE_DATE<:until", timeBindVariableList);
  fwccQuery->addToOrderList("CHANGE_DATE");
  coral::ICursor& fwccCursor = fwccQuery->execute();
  int nrows = 0;
  while (fwccCursor.next()) {
    nrows++;
    const coral::Attribute& dp = fwccCursor.currentRow()["DPID"];
    const coral::Attribute& vmon = fwccCursor.currentRow()["ACTUAL_VMON"];
    const coral::Attribute& imon = fwccCursor.currentRow()["ACTUAL_IMON"];
    coral::TimeStamp changeTime = fwccCursor.currentRow()["CHANGE_DATE"].data<coral::TimeStamp>();
    long long int cTimeValue = changeTime.total_nanoseconds();
    if (!copyOffline)
      cTimeValue = -cTimeValue;
    if (dp.isNull()) {
      std::cout << "------- " << nrows << std::endl;
      continue;
    }
    int dpId = 10 * static_cast<int>(0.01 + fwccCursor.currentRow()["DPID"].data<float>());
    if (!(vmon.isNull())) {
      while (periodBuffer.find(cTimeValue) != periodBuffer.end())
        cTimeValue++;
      int chan = dpId + 1;
      periodBuffer.insert(std::pair<long long int, channelValue>(cTimeValue, channelValue(chan, vmon.data<float>())));
    }
    if (!(imon.isNull())) {
      while (periodBuffer.find(cTimeValue) != periodBuffer.end())
        cTimeValue++;
      int chan = dpId + 2;
      periodBuffer.insert(std::pair<long long int, channelValue>(cTimeValue, channelValue(chan, imon.data<float>())));
    }
  }

  long long int dTime = minTime;
  dTime <<= 32;
  std::cout << "data found in period: " << periodBuffer.size() << std::endl;
  std::map<long long int, channelValue>::const_iterator bufIter = periodBuffer.begin();
  std::map<long long int, channelValue>::const_iterator bufIend = periodBuffer.end();

  bool changedStatus = false;
  while (bufIter != bufIend) {
    const std::pair<long long int, channelValue>& entry = *bufIter++;
    long long int mTime = entry.first;
    if (!copyOffline)
      mTime = -mTime;
    channelValue cValue = entry.second;
    int chan = cValue.first;
    float cont = cValue.second;
    mapIter = snapshotValues.find(chan);
    if ((mapIter != mapIend) && (mapIter->second.first < mTime)) {
      nextFound = condTime(mTime);
      if (changedStatus) {
        if (nextFound > timeLimit) {
          DTHVStatus* hvStatus = offlineList();
          std::cout << "new payload " << hvStatus->end() - hvStatus->begin() << std::endl;
          tmpContainer.push_back(std::make_pair(hvStatus, lastFound));
          changedStatus = false;
          if (!(--maxPayload)) {
            procUntil = lastFound;
            std::cout << "max payload number reached" << std::endl;
            break;
          }
        }
      }
      if (copyOffline && !changedStatus && checkStatusChange(chan, mapIter->second.second, cont)) {
        timeLimit = nextFound + dTime;
        changedStatus = true;
      }
      mapIter->second = timedMeasurement(lastStamp = mTime, cont);
      lastFound = nextFound;
      missingChannels--;
    }
  }

  std::cout << nrows << std::endl;
  return nrows;
}

void DTHVStatusHandler::copyHVData() {
  long long int dTime = fwdTime;
  dTime <<= 32;

  cond::Time_t condSince = procSince;
  cond::Time_t condUntil = condSince + dTime;
  if (condUntil > procUntil)
    condUntil = procUntil;

  int dum = 0;
  lastStatus = nullptr;
  while (condSince < condUntil) {
    checkForPeriod(condSince, condUntil, dum, true);
    condSince = condUntil;
    condUntil = condSince + dTime;
    if (condUntil > procUntil)
      condUntil = procUntil;
  }
  std::cout << "call filterData " << std::endl;
  filterData();
  std::cout << "filterData return " << switchOff << " " << lastFound << " " << maxPayload << " " << m_to_transfer.size()
            << std::endl;
  if (switchOff || ((lastFound != 0) && (maxPayload > 0))) {
    DTHVStatus* hvStatus = offlineList();
    m_to_transfer.push_back(std::make_pair(hvStatus, lastFound));
  }

  return;
}

DTHVStatus* DTHVStatusHandler::offlineList() {
  DTHVStatus* hv = new DTHVStatus(dataTag);
  int type;
  float valueA = 0.0;
  float valueL = 0.0;
  float valueR = 0.0;
  float valueS = 0.0;
  float valueC = 0.0;
  std::map<int, int>::const_iterator layerIter = layerMap.begin();
  std::map<int, int>::const_iterator layerIend = layerMap.end();
  while (layerIter != layerIend) {
    const std::pair<int, int>& chanEntry = *layerIter++;
    int rawId = chanEntry.first;
    DTWireId chlId(rawId);
    int whe = chlId.wheel();
    int sta = chlId.station();
    int sec = chlId.sector();
    int qua = chlId.superLayer();
    int lay = chlId.layer();
    int l_p = chlId.wire();
    if (l_p != 10)
      continue;
    for (type = 1; type <= 2; type++) {
      getLayerValues(rawId, type, valueL, valueR, valueS, valueC);
      for (l_p = 0; l_p <= 1; l_p++) {
        int rPart = layerId(rawId, l_p).rawId();
        switch (l_p) {
          case 0:
            valueA = valueL;
            break;
          case 1:
            valueA = valueR;
            break;
          default:
            break;
        }
        //  std::cout << "layer values: " << type << " " << valueA << " "
        //                                               << valueS << " "
        //                                               << valueC << std::endl;
        DTHVAbstractCheck::flag flag =
            hvChecker->checkCurrentStatus(rPart, type, valueA, valueC, valueS, snapshotValues, aliasMap, layerMap);
        if (!flag.a && !flag.c && !flag.s)
          continue;
        setChannelFlag(hv, whe, sta, sec, qua, lay, l_p, flag);
        std::map<int, std::vector<int>*>::const_iterator m_iter = channelSplit.find(rPart);
        std::map<int, std::vector<int>*>::const_iterator m_iend = channelSplit.end();
        if (m_iter != m_iend) {
          std::vector<int>* cList = m_iter->second;
          std::vector<int>::const_iterator l_iter = cList->begin();
          std::vector<int>::const_iterator l_iend = cList->end();
          while (l_iter != l_iend) {
            DTWireId chlId(*l_iter++);
            int wh2 = chlId.wheel();
            int st2 = chlId.station();
            int se2 = chlId.sector();
            int qu2 = chlId.superLayer();
            int la2 = chlId.layer();
            int lp2 = chlId.wire() - 10;
            //	    std::cout << "duplicate "
            //                      << whe << " " << sta << " " << sec << " "
            //                      << qua << " " << lay << " " << l_p << " ---> "
            //                      << wh2 << " " << st2 << " " << se2 << " "
            //                      << qu2 << " " << la2 << " " << lp2 << std::endl;
            setChannelFlag(hv, wh2, st2, se2, qu2, la2, lp2, flag);
          }
        }
      }
    }
  }
  return hv;
}

void DTHVStatusHandler::getLayerValues(int rawId, int type, float& valueL, float& valueR, float& valueS, float& valueC) {
  valueL = valueR = valueS = valueC = 0.0;
  DTWireId chlId(rawId);
  std::map<int, timedMeasurement>::const_iterator snapIter = snapshotValues.begin();
  std::map<int, timedMeasurement>::const_iterator snapIend = snapshotValues.end();
  int rawL = layerId(rawId, 0).rawId();
  int rawR = layerId(rawId, 1).rawId();
  int rawS = layerId(rawId, 2).rawId();
  int rawC = layerId(rawId, 3).rawId();
  std::map<int, int>::const_iterator layerIter;
  std::map<int, int>::const_iterator layerIend = layerMap.end();
  if ((layerIter = layerMap.find(rawL)) != layerIend) {
    const std::pair<int, int>& layerEntry = *layerIter;
    int dpId = layerEntry.second;
    snapIter = snapshotValues.find((dpId * 10) + type);
    if (snapIter != snapIend) {
      const std::pair<int, timedMeasurement>& snapEntry = *snapIter;
      valueL = snapEntry.second.second;
    } else
      std::cout << "snapR not found" << std::endl;
  } else
    std::cout << "rawR not found" << std::endl;
  if ((layerIter = layerMap.find(rawR)) != layerIend) {
    const std::pair<int, int>& layerEntry = *layerIter;
    int dpId = layerEntry.second;
    snapIter = snapshotValues.find((dpId * 10) + type);
    if (snapIter != snapIend) {
      const std::pair<int, timedMeasurement>& snapEntry = *snapIter;
      valueR = snapEntry.second.second;
    } else
      std::cout << "snapL not found" << std::endl;
  } else
    std::cout << "rawL not found" << std::endl;
  if ((layerIter = layerMap.find(rawS)) != layerIend) {
    const std::pair<int, int>& layerEntry = *layerIter;
    int dpId = layerEntry.second;
    snapIter = snapshotValues.find((dpId * 10) + type);
    if (snapIter != snapIend) {
      const std::pair<int, timedMeasurement>& snapEntry = *snapIter;
      valueS = snapEntry.second.second;
    } else
      std::cout << "snapS not found" << std::endl;
  } else
    std::cout << "rawS not found" << std::endl;
  if ((layerIter = layerMap.find(rawC)) != layerIend) {
    const std::pair<int, int>& layerEntry = *layerIter;
    int dpId = layerEntry.second;
    snapIter = snapshotValues.find((dpId * 10) + type);
    if (snapIter != snapIend) {
      const std::pair<int, timedMeasurement>& snapEntry = *snapIter;
      valueC = snapEntry.second.second;
    } else
      std::cout << "snapC not found" << std::endl;
  } else
    std::cout << "rawC not found" << std::endl;
  //  std::cout << "layer values... " << type << " " << valueL << " "
  //                                                 << valueR << " "
  //                                                 << valueS << " "
  //                                                 << valueC << std::endl;
  return;
}

void DTHVStatusHandler::setChannelFlag(
    DTHVStatus* hv, int whe, int sta, int sec, int qua, int lay, int l_p, const DTHVAbstractCheck::flag& flag) {
  int fCell = 0;
  int lCell = 99;
  int flagA = 0;
  int flagC = 0;
  int flagS = 0;
  int searchStatus = hv->get(whe, sta, sec, qua, lay, l_p, fCell, lCell, flagA, flagC, flagS);
  if (searchStatus) {
    DTWireId wireId(whe, sta, sec, qua, lay, 10 + l_p);
    std::map<int, int>::const_iterator splitIter = laySplit.find(wireId.rawId());
    std::map<int, int>::const_iterator splitIend = laySplit.end();
    if (splitIter != splitIend) {
      int code = splitIter->second;
      fCell = code / 10000;
      lCell = code % 10000;
    }
  }
  flagA |= flag.a;
  flagC |= flag.c;
  flagS |= flag.s;
  hv->set(whe, sta, sec, qua, lay, l_p, fCell, lCell, flagA, flagC, flagS);
  return;
}

int DTHVStatusHandler::checkStatusChange(int chan, float oldValue, float newValue) {
  int dpId = chan / 10;
  int type = chan % 10;
  std::map<int, int>::const_iterator aliasIter = aliasMap.find(dpId);
  std::map<int, int>::const_iterator aliasIend = aliasMap.end();
  if (aliasIter == aliasIend)
    return false;
  int rawId = aliasIter->second;
  DTWireId chlId(rawId);
  int l_p = chlId.wire();
  float valueL = 0.0;
  float valueR = 0.0;
  float valueS = 0.0;
  float valueC = 0.0;
  getLayerValues(rawId, type, valueL, valueR, valueS, valueC);
  //  std::cout << "layer values: " << type << " " << valueL << " "
  //                                               << valueR << " "
  //                                               << valueS << " "
  //                                               << valueC << std::endl;
  DTHVAbstractCheck::flag oldStatusL = hvChecker->checkCurrentStatus(
      layerId(rawId, 0).rawId(), type, valueL, valueC, valueS, snapshotValues, aliasMap, layerMap);
  DTHVAbstractCheck::flag oldStatusR = hvChecker->checkCurrentStatus(
      layerId(rawId, 1).rawId(), type, valueR, valueC, valueS, snapshotValues, aliasMap, layerMap);
  switch (l_p) {
    case 10:
      if (valueL != oldValue)
        std::cout << "*** INCONSISTENT DATA!!!!! " << type << " " << l_p << " " << oldValue << " " << valueL << " "
                  << std::endl;
      valueL = newValue;
      break;
    case 11:
      if (valueR != oldValue)
        std::cout << "*** INCONSISTENT DATA!!!!! " << type << " " << l_p << " " << oldValue << " " << valueR << " "
                  << std::endl;
      valueR = newValue;
      break;
    case 12:
      if (valueS != oldValue)
        std::cout << "*** INCONSISTENT DATA!!!!! " << type << " " << l_p << " " << oldValue << " " << valueS << " "
                  << std::endl;
      valueS = newValue;
      break;
    case 13:
      if (valueC != oldValue)
        std::cout << "*** INCONSISTENT DATA!!!!! " << type << " " << l_p << " " << oldValue << " " << valueC << " "
                  << std::endl;
      valueC = newValue;
      break;
    default:
      break;
  }
  DTHVAbstractCheck::flag newStatusL = hvChecker->checkCurrentStatus(
      layerId(rawId, 0).rawId(), type, valueL, valueC, valueS, snapshotValues, aliasMap, layerMap);
  DTHVAbstractCheck::flag newStatusR = hvChecker->checkCurrentStatus(
      layerId(rawId, 1).rawId(), type, valueR, valueC, valueS, snapshotValues, aliasMap, layerMap);

  if (DTHVAbstractCheck::compare(newStatusL, oldStatusL) && DTHVAbstractCheck::compare(newStatusR, oldStatusR))
    return 0;
  std::cout << "changed status: " << chan << " from " << oldValue << " to " << newValue << std::endl;
  return 1;
}

void DTHVStatusHandler::filterData() {
  int maxTime = 100;
  int maxTtot = 600;
  int minDiff = 88;

  int iTime = 0;
  int pTime = 0;
  int nTime = 0;
  int iSize;
  int pSize;
  int nSize;

  std::vector<std::pair<DTHVStatus*, cond::Time_t> >::const_iterator iter = tmpContainer.begin();
  std::vector<std::pair<DTHVStatus*, cond::Time_t> >::const_iterator iend = tmpContainer.end();
  std::vector<std::pair<DTHVStatus*, cond::Time_t> >::const_iterator prev;
  std::vector<std::pair<DTHVStatus*, cond::Time_t> >::const_iterator next;

  while (iter != iend) {
    switchOff = false;
    next = iter;
    prev = next++;
    if (next == iend)
      next = prev;
    const DTHVStatus* iPtr = iter->first;
    const DTHVStatus* pPtr = prev->first;
    const DTHVStatus* nPtr = next->first;
    iSize = std::distance(iPtr->begin(), iPtr->end());
    pSize = std::distance(pPtr->begin(), pPtr->end());
    nSize = std::distance(nPtr->begin(), nPtr->end());
    int dtot = nSize - pSize;
    prev = next;
    while (++next != iend) {
      pPtr = prev->first;
      nPtr = next->first;
      pSize = std::distance(pPtr->begin(), pPtr->end());
      nSize = std::distance(nPtr->begin(), nPtr->end());
      int diff = nSize - pSize;
      iTime = static_cast<int>((iter->second >> 32) & 0xffffffff);
      pTime = static_cast<int>((prev->second >> 32) & 0xffffffff);
      nTime = static_cast<int>((next->second >> 32) & 0xffffffff);
      if ((nTime - pTime) > maxTime)
        break;
      if ((nTime - iTime) > maxTtot)
        break;
      if ((dtot * diff) < 0)
        break;
      prev = next;
    }
    pPtr = prev->first;
    iSize = std::distance(iPtr->begin(), iPtr->end());
    pSize = std::distance(pPtr->begin(), pPtr->end());
    dtot = pSize - iSize;
    int dist = pTime - iTime;
    if ((dtot < -minDiff) && (dist < maxTtot)) {
      std::cout << "  ******** SWITCH ON " << std::distance(iter, prev) << " " << iTime << " " << pTime << " " << iSize
                << " " << pSize << std::endl;
      m_to_transfer.push_back(std::make_pair(prev->first, prev->second));
      while (iter != prev)
        delete (iter++->first);
    }
    if ((dtot > minDiff) && (dist < maxTtot)) {
      std::cout << "  ******** SWITCH OFF " << std::distance(iter, prev) << " " << iTime << " " << pTime << " " << iSize
                << " " << pSize << std::endl;
      m_to_transfer.push_back(std::make_pair(prev->first, iter->second));
      switchOff = true;
      while (iter != prev)
        delete (iter++->first);
    }
    if (((dtot >= -minDiff) && (dtot <= minDiff)) || (dist >= maxTtot)) {
      while (iter != next) {
        const std::pair<DTHVStatus*, cond::Time_t>& entry = *iter++;
        m_to_transfer.push_back(std::make_pair(entry.first, entry.second));
      }
    }
    iter = next;
  }
}

DTWireId DTHVStatusHandler::layerId(int rawId, int l_p) {
  DTWireId chlId(rawId);
  int whe = chlId.wheel();
  int sta = chlId.station();
  int sec = chlId.sector();
  int qua = chlId.superLayer();
  int lay = chlId.layer();
  DTWireId chl(whe, sta, sec, qua, lay, 10 + l_p);
  return chl;
}

coral::TimeStamp DTHVStatusHandler::coralTime(const cond::Time_t& time) {
  long long int iTime = ((((time >> 32) & 0xFFFFFFFF) * 1000000000) + ((time & 0xFFFFFFFF) * 1000));
  coral::TimeStamp cTime(iTime);
  return cTime;
}

cond::Time_t DTHVStatusHandler::condTime(const coral::TimeStamp& time) {
  cond::Time_t cTime =
      ((time.total_nanoseconds() / 1000000000) << 32) + ((time.total_nanoseconds() % 1000000000) / 1000);
  return cTime;
}

cond::Time_t DTHVStatusHandler::condTime(long long int time) {
  cond::Time_t cTime = ((time / 1000000000) << 32) + ((time % 1000000000) / 1000);
  return cTime;
}

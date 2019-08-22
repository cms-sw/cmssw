#include <stdexcept>
#include <string>
#include <cmath>

#include "OnlineDB/EcalCondDB/interface/RunDCSLVDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

RunDCSLVDat::RunDCSLVDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_lv = 0;
  m_lvnom = 0;
  m_status = 0;
  m_tstatus = 0;
}

RunDCSLVDat::~RunDCSLVDat() {}

void RunDCSLVDat::prepareWrite() noexcept(false) {}

void RunDCSLVDat::writeDB(const EcalLogicID* ecid, const RunDCSLVDat* item, RunIOV* iov) noexcept(false) {}

void RunDCSLVDat::fetchData(map<EcalLogicID, RunDCSLVDat>* fillMap, RunIOV* iov) noexcept(false) {
  fetchLastData(fillMap);
}

ResultSet* RunDCSLVDat::getBarrelRset() {
  ResultSet* rset = nullptr;
  string query =
      "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
      " d.value_number , '5' NOMINAL_VALUE , d.VALUE_TIMESTAMP "
      "FROM " +
      getEBAccount() +
      ".FWWIENERMARATHONCHANNEL_LV d "
      " JOIN " +
      getEBAccount() +
      ".LV_MAPPING h on "
      " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name and "
      "dpe_name='MEASUREMENTSENSEVOLTAGE' ";
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSLVDat::getBarrelRset():  ") + e.getMessage() + " " + query));
#else
    throw(std::runtime_error(std::string("RunDCSLVDat::getBarrelRset():  error code ") +
                             std::to_string(e.getErrorCode()) + " " + query));
#endif
  }
  return rset;
}

ResultSet* RunDCSLVDat::getEndcapRset() {
  ResultSet* rset = nullptr;
  string query =
      "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
      " d.VALUE_NUMBER, '5' NOMINAL_VALUE , d.VALUE_TIMESTAMP "
      "FROM " +
      getEEAccount() +
      ".FWWIENERMARATHONCHANNEL_LV d "
      " JOIN " +
      getEEAccount() +
      ".EE_LV_MAPPING h on "
      " h.DPID = d.DPID join channelview cv on cv.logic_id=h.logic_id WHERE cv.maps_to = cv.name and "
      "dpe_name='MEASUREMENTSENSEVOLTAGE' ";
  try {
    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSLVDat::getEndcapRset():  ") + e.getMessage() + " " + query));
#else
    throw(std::runtime_error(std::string("RunDCSLVDat::getEndcapRset():  error code ") +
                             std::to_string(e.getErrorCode()) + " " + query));
#endif
  }
  return rset;
}

void RunDCSLVDat::fillTheMap(ResultSet* rset, map<EcalLogicID, RunDCSLVDat>* fillMap) {
  std::pair<EcalLogicID, RunDCSLVDat> p;
  RunDCSLVDat dat;
  DateHandler dh(m_env, m_conn);

  try {
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setLV(rset->getFloat(7));
      dat.setLVNominal(rset->getFloat(8));
      Date sinceDate = rset->getDate(9);
      Tm sinceTm = dh.dateToTm(sinceDate);
      dat.setStatus(0);
      if (p.first.getName() == "EB_LV_channel") {
        setStatusForBarrel(dat, sinceTm);
      } else {
        setStatusForEndcaps(dat, sinceTm);
      }
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSLVDat::fetchData():  ") + e.getMessage()));
#else
    throw(std::runtime_error(std::string("RunDCSLVDat::fetchData():  error code ") + std::to_string(e.getErrorCode())));
#endif
  }
}

int RunDCSLVDat::nowMicroseconds() {
  Tm t_now_gmt;

  t_now_gmt.setToCurrentGMTime();
  int t_now_gmt_micros = t_now_gmt.microsTime();
  return t_now_gmt_micros;
}

void RunDCSLVDat::setStatusForBarrel(RunDCSLVDat& dat, const Tm& sinceTm) {
  int t_now_gmt_micros = nowMicroseconds();

  if (fabs(dat.getLV() - dat.getLVNominal()) * 1000 > maxLVDifferenceEB) {
    dat.setStatus(LVNOTNOMINAL);
  }
  if (dat.getLV() * 1000 < minLV) {
    dat.setStatus(LVOFF);
  }

  int result = 0;
  int d = ((int)t_now_gmt_micros - (int)sinceTm.microsTime());
  if (d > maxDifference) {
    result = -d / 1000000;
  }
  dat.setTimeStatus(result);
}

void RunDCSLVDat::setStatusForEndcaps(RunDCSLVDat& dat, const Tm& sinceTm) {
  int t_now_gmt_micros = nowMicroseconds();

  if (fabs(dat.getLV() - dat.getLVNominal()) * 1000 > maxLVDifferenceEE) {
    dat.setStatus(LVNOTNOMINAL);
  }
  if (dat.getLV() * 1000 < minLV) {
    dat.setStatus(LVOFF);
  }

  int result = 0;
  int d = ((int)t_now_gmt_micros - (int)sinceTm.microsTime());
  if (d > maxDifference) {
    result = -d / 1000000;
  }
  dat.setTimeStatus(result);
}

void RunDCSLVDat::fetchLastData(map<EcalLogicID, RunDCSLVDat>* fillMap) noexcept(false) {
  this->checkConnection();

  fillMap->clear();

  try {
    std::pair<EcalLogicID, RunDCSLVDat> p;
    RunDCSLVDat dat;

    ResultSet* rset = getBarrelRset();

    fillTheMap(rset, fillMap);
    rset = getEndcapRset();

    fillTheMap(rset, fillMap);
  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSLVDat::fetchData():  ") + e.getMessage()));
#else
    throw(std::runtime_error(std::string("RunDCSLVDat::fetchData():  error code ") + std::to_string(e.getErrorCode())));
#endif
  }
}

#include <stdexcept>
#include <string>
#include <cmath>
#include <list>
#include <string>
#include <map>

#include "OnlineDB/EcalCondDB/interface/RunDCSMagnetDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

using namespace std;
using namespace oracle::occi;

RunDCSMagnetDat::RunDCSMagnetDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_current = 0;
  m_time = Tm();
}

RunDCSMagnetDat::~RunDCSMagnetDat() {}

void RunDCSMagnetDat::setTime(const Tm& start) { m_time = start; }

Tm RunDCSMagnetDat::getTime() const { return m_time; }

void RunDCSMagnetDat::prepareWrite() noexcept(false) {}

void RunDCSMagnetDat::writeDB(const EcalLogicID* ecid, const RunDCSMagnetDat* item, RunIOV* iov) noexcept(false) {}

void RunDCSMagnetDat::fetchData(map<EcalLogicID, RunDCSMagnetDat>* fillMap, RunIOV* iov) noexcept(false) {
  std::cout << "going to call fetchLastData" << std::endl;
  fetchLastData(fillMap);
  std::cout << "returned from fetchLastData" << std::endl;
}

ResultSet* RunDCSMagnetDat::getMagnetRset() {
  DateHandler dh(m_env, m_conn);

  ResultSet* rset = nullptr;
  string query = "SELECT c.name, c.logic_id, c.id1, c.id2, c.id3, c.maps_to , v.value_number, v.change_date from " +
                 getMagnetAccount() +
                 ".CMSFWMAGNET_LV v, channelview c where v.dpe_name= 'CURRENT' and  c.name=maps_to and c.name='EB' ";
  try {
    std::cout << "query:" << query << std::endl;

    m_readStmt->setSQL(query);
    rset = m_readStmt->executeQuery();
  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSMagnetDat::getBarrelRset():  ") + e.getMessage() + " " + query));
#else
    throw(std::runtime_error(std::string("RunDCSMagnetDat::getBarrelRset():  error code ") +
                             std::to_string(e.getErrorCode()) + " " + query));
#endif
  }
  return rset;
}

void RunDCSMagnetDat::fillTheMap(ResultSet* rset, map<EcalLogicID, RunDCSMagnetDat>* fillMap) {
  // method for last value queries

  std::pair<EcalLogicID, RunDCSMagnetDat> p;
  RunDCSMagnetDat dat;
  DateHandler dh(m_env, m_conn);

  try {
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      std::cout << "done the logic id" << std::endl;
      dat.setMagnetCurrent(rset->getFloat(7));
      std::cout << "done the magnet current" << std::endl;

      Date sinceDate = rset->getDate(8);
      std::cout << "done the date" << std::endl;

      Tm sinceTm = dh.dateToTm(sinceDate);
      dat.setTime(sinceTm);

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSMagnetDat::fetchData():  ") + e.getMessage()));
#else
    throw(std::runtime_error(std::string("RunDCSMagnetDat::fetchData():  error code ") +
                             std::to_string(e.getErrorCode())));
#endif
  }
}

int RunDCSMagnetDat::nowMicroseconds() {
  Tm t_now_gmt;

  t_now_gmt.setToCurrentGMTime();
  int t_now_gmt_micros = t_now_gmt.microsTime();
  return t_now_gmt_micros;
}

void RunDCSMagnetDat::fetchLastData(map<EcalLogicID, RunDCSMagnetDat>* fillMap) noexcept(false) {
  this->checkConnection();

  std::cout << "fetchLastData>>1" << std::endl;

  fillMap->clear();

  std::cout << "fetchLastData>>2" << std::endl;

  try {
    std::pair<EcalLogicID, RunDCSMagnetDat> p;
    RunDCSMagnetDat dat;
    std::cout << "fetchLastData>>3" << std::endl;

    ResultSet* rset = getMagnetRset();

    std::cout << "fetchLastData>>4" << std::endl;

    fillTheMap(rset, fillMap);
    std::cout << "fetchLastData>>5" << std::endl;

  } catch (SQLException& e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    throw(std::runtime_error(std::string("RunDCSMagnetDat::fetchData():  ") + e.getMessage()));
#else
    throw(std::runtime_error(std::string("RunDCSMagnetDat::fetchData():  error code ") +
                             std::to_string(e.getErrorCode())));
#endif
  }
}

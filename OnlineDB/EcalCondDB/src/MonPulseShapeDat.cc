#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonPulseShapeDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPulseShapeDat::MonPulseShapeDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_samplesG1.assign(10, 0);
  m_samplesG6.assign(10, 0);
  m_samplesG12.assign(10, 0);
}

MonPulseShapeDat::~MonPulseShapeDat() {}

void MonPulseShapeDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO mon_pulse_shape_dat (iov_id, logic_id, "
        "g1_avg_sample_01, g1_avg_sample_02, g1_avg_sample_03, g1_avg_sample_04, g1_avg_sample_05, g1_avg_sample_06, "
        "g1_avg_sample_07, g1_avg_sample_08, g1_avg_sample_09, g1_avg_sample_10, "
        "g6_avg_sample_01, g6_avg_sample_02, g6_avg_sample_03, g6_avg_sample_04, g6_avg_sample_05, g6_avg_sample_06, "
        "g6_avg_sample_07, g6_avg_sample_08, g6_avg_sample_09, g6_avg_sample_10, "
        "g12_avg_sample_01, g12_avg_sample_02, g12_avg_sample_03, g12_avg_sample_04, g12_avg_sample_05, "
        "g12_avg_sample_06, g12_avg_sample_07, g12_avg_sample_08, g12_avg_sample_09, g12_avg_sample_10) "
        "VALUES (:iov_id, :logic_id, "
        ":g1_avg_sample_01, :g1_avg_sample_02, :g1_avg_sample_03, :g1_avg_sample_04, :g1_avg_sample_05, "
        ":g1_avg_sample_06, :g1_avg_sample_07, :g1_avg_sample_08, :g1_avg_sample_09, :g1_avg_sample_10,"
        ":g6_avg_sample_01, :g6_avg_sample_02, :g6_avg_sample_03, :g6_avg_sample_04, :g6_avg_sample_05, "
        ":g6_avg_sample_06, :g6_avg_sample_07, :g6_avg_sample_08, :g6_avg_sample_09, :g6_avg_sample_10,"
        ":g12_avg_sample_01, :g12_avg_sample_02, :g12_avg_sample_03, :g12_avg_sample_04, :g12_avg_sample_05, "
        ":g12_avg_sample_06, :g12_avg_sample_07, :g12_avg_sample_08, :g12_avg_sample_09, :g12_avg_sample_10)");

  } catch (SQLException& e) {
    throw(std::runtime_error("MonPulseShapeDat::prepareWrite():  " + e.getMessage()));
  }
}

void MonPulseShapeDat::writeDB(const EcalLogicID* ecid, const MonPulseShapeDat* item, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonPulseShapeDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MonPulseShapeDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    int gain[] = {1, 6, 12};
    std::vector<float> samples;
    for (int i = 0; i < 3; i++) {
      samples = item->getSamples(gain[i]);
      for (int j = 0; j < 10; j++) {
        m_writeStmt->setFloat(3 + (10 * i) + j, samples.at(j));
      }
    }

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MonPulseShapeDat::writeDB:  " + e.getMessage()));
  } catch (exception& e) {
    throw(std::runtime_error("MonPulseShapeDat::writeDB:  " + string(e.what())));
  }
}

void MonPulseShapeDat::fetchData(std::map<EcalLogicID, MonPulseShapeDat>* fillMap, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MonPulseShapeDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.g1_avg_sample_01, d.g1_avg_sample_02, d.g1_avg_sample_03, d.g1_avg_sample_04, d.g1_avg_sample_05, "
        "d.g1_avg_sample_06, d.g1_avg_sample_07, d.g1_avg_sample_08, d.g1_avg_sample_09, d.g1_avg_sample_10, "
        "d.g6_avg_sample_01, d.g6_avg_sample_02, d.g6_avg_sample_03, d.g6_avg_sample_04, d.g6_avg_sample_05, "
        "d.g6_avg_sample_06, d.g6_avg_sample_07, d.g6_avg_sample_08, d.g6_avg_sample_09, d.g6_avg_sample_10, "
        "d.g12_avg_sample_01, d.g12_avg_sample_02, d.g12_avg_sample_03, d.g12_avg_sample_04, d.g12_avg_sample_05, "
        "d.g12_avg_sample_06, d.g12_avg_sample_07, d.g12_avg_sample_08, d.g12_avg_sample_09, d.g12_avg_sample_10 "
        "FROM channelview cv JOIN mon_pulse_shape_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MonPulseShapeDat> p;
    MonPulseShapeDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      int gain[] = {1, 6, 12};
      std::vector<float> samples(10);
      for (int i = 0; i < 3; i++) {
        samples.clear();
        for (int j = 0; j < 10; j++) {
          samples.push_back(rset->getFloat(7 + (10 * i) + j));
        }
        dat.setSamples(samples, gain[i]);
      }

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MonPulseShapeDat::fetchData:  " + e.getMessage()));
  }
}

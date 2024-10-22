#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonDelaysTTDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

MonDelaysTTDat::MonDelaysTTDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_delayMean = 0;
  m_delayRMS = 0;
  m_taskStatus = false;
}

MonDelaysTTDat::~MonDelaysTTDat() {}

void MonDelaysTTDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO mon_delays_tt_dat (iov_id, logic_id, "
        "delay_mean, delay_rms, task_status) "
        "VALUES (:iov_id, :logic_id, "
        ":delay_mean, :delay_rms, :task_status)");
  } catch (SQLException& e) {
    throw(std::runtime_error("MonDelaysTTDat::prepareWrite():  " + e.getMessage()));
  }
}

void MonDelaysTTDat::writeDB(const EcalLogicID* ecid, const MonDelaysTTDat* item, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonDelaysTTDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MonDelaysTTDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getDelayMean());
    m_writeStmt->setFloat(4, item->getDelayRMS());
    m_writeStmt->setInt(5, item->getTaskStatus());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MonDelaysTTDat::writeDB():  " + e.getMessage()));
  }
}

void MonDelaysTTDat::fetchData(std::map<EcalLogicID, MonDelaysTTDat>* fillMap, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MonDelaysTTDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.delay_mean, d.delay_rms, d.task_status "
        "FROM channelview cv JOIN mon_delays_tt_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MonDelaysTTDat> p;
    MonDelaysTTDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setDelayMean(rset->getFloat(7));
      dat.setDelayRMS(rset->getFloat(8));
      dat.setTaskStatus(rset->getInt(9));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MonDelaysTTDat::fetchData():  " + e.getMessage()));
  }
}

void MonDelaysTTDat::writeArrayDB(const std::map<EcalLogicID, MonDelaysTTDat>* data, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonDelaysTTDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  float* xx = new float[nrows];
  float* yy = new float[nrows];
  int* st = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* st_len = new ub2[nrows];

  const EcalLogicID* channel;
  const MonDelaysTTDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, MonDelaysTTDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("MonDelaysTTDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    float x = dataitem->getDelayMean();
    float y = dataitem->getDelayRMS();
    int statu = dataitem->getTaskStatus();

    xx[count] = x;
    yy[count] = y;
    st[count] = statu;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    st_len[count] = sizeof(st[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)st, OCCIINT, sizeof(st[0]), st_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx;
    delete[] yy;
    delete[] st;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;
    delete[] y_len;
    delete[] st_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("MonDelaysTTDat::writeArrayDB():  " + e.getMessage()));
  }
}

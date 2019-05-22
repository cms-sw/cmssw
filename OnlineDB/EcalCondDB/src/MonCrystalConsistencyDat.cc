#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonCrystalConsistencyDat.h"

using namespace std;
using namespace oracle::occi;

MonCrystalConsistencyDat::MonCrystalConsistencyDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_processedEvents = 0;
  m_problematicEvents = 0;
  m_problemsID = 0;
  m_problemsGainZero = 0;
  m_problemsGainSwitch = 0;
  m_taskStatus = false;
}

MonCrystalConsistencyDat::~MonCrystalConsistencyDat() {}

void MonCrystalConsistencyDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO mon_crystal_consistency_dat (iov_id, logic_id, "
        "processed_events, problematic_events, problems_id, problems_gain_zero, problems_gain_switch, task_status) "
        "VALUES (:iov_id, :logic_id, "
        ":3, :4, :5, :6, :7, :8)");
  } catch (SQLException& e) {
    throw(std::runtime_error("MonCrystalConsistencyDat::prepareWrite():  " + e.getMessage()));
  }
}

void MonCrystalConsistencyDat::writeDB(const EcalLogicID* ecid,
                                       const MonCrystalConsistencyDat* item,
                                       MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonCrystalConsistencyDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MonCrystalConsistencyDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getProcessedEvents());
    m_writeStmt->setInt(4, item->getProblematicEvents());
    m_writeStmt->setInt(5, item->getProblemsID());
    m_writeStmt->setInt(6, item->getProblemsGainZero());
    m_writeStmt->setInt(7, item->getProblemsGainSwitch());
    m_writeStmt->setInt(8, item->getTaskStatus());
    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MonCrystalConsistencyDat::writeDB():  " + e.getMessage()));
  }
}

void MonCrystalConsistencyDat::fetchData(std::map<EcalLogicID, MonCrystalConsistencyDat>* fillMap,
                                         MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MonCrystalConsistencyDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.processed_events, d.problematic_events, d.problems_id, d.problems_gain_zero, d.problems_gain_switch, "
        "d.task_status "
        "FROM channelview cv JOIN mon_crystal_consistency_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MonCrystalConsistencyDat> p;
    MonCrystalConsistencyDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setProcessedEvents(rset->getInt(7));
      dat.setProblematicEvents(rset->getInt(8));
      dat.setProblemsID(rset->getInt(9));
      dat.setProblemsGainZero(rset->getInt(10));
      dat.setProblemsGainSwitch(rset->getInt(11));
      dat.setTaskStatus(rset->getInt(12));

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException& e) {
    throw(std::runtime_error("MonCrystalConsistencyDat::fetchData():  " + e.getMessage()));
  }
}

void MonCrystalConsistencyDat::writeArrayDB(const std::map<EcalLogicID, MonCrystalConsistencyDat>* data,
                                            MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonCrystalConsistencyDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  int* xx = new int[nrows];
  int* yy = new int[nrows];
  int* zz = new int[nrows];
  int* ww = new int[nrows];
  int* uu = new int[nrows];
  int* st = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* z_len = new ub2[nrows];
  ub2* w_len = new ub2[nrows];
  ub2* u_len = new ub2[nrows];
  ub2* st_len = new ub2[nrows];

  const EcalLogicID* channel;
  const MonCrystalConsistencyDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, MonCrystalConsistencyDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("MonCrystalConsistencyDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    int x = dataitem->getProcessedEvents();
    int y = dataitem->getProblematicEvents();
    int z = dataitem->getProblemsID();
    int w = dataitem->getProblemsGainZero();
    int u = dataitem->getProblemsGainSwitch();
    int statu = dataitem->getTaskStatus();

    xx[count] = x;
    yy[count] = y;
    zz[count] = z;
    ww[count] = w;
    uu[count] = u;
    st[count] = statu;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    z_len[count] = sizeof(zz[count]);
    w_len[count] = sizeof(ww[count]);
    u_len[count] = sizeof(uu[count]);
    st_len[count] = sizeof(st[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIINT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIINT, sizeof(zz[0]), z_len);
    m_writeStmt->setDataBuffer(6, (dvoid*)ww, OCCIINT, sizeof(ww[0]), w_len);
    m_writeStmt->setDataBuffer(7, (dvoid*)uu, OCCIINT, sizeof(uu[0]), u_len);
    m_writeStmt->setDataBuffer(8, (dvoid*)st, OCCIINT, sizeof(st[0]), st_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx;
    delete[] yy;
    delete[] zz;
    delete[] ww;
    delete[] uu;
    delete[] st;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;
    delete[] y_len;
    delete[] z_len;
    delete[] w_len;
    delete[] u_len;
    delete[] st_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("MonCrystalConsistencyDat::writeArrayDB():  " + e.getMessage()));
  }
}

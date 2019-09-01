#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigTimingDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigTimingInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigTimingDat::FEConfigTimingDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_par1 = 0;
  m_par2 = 0;
}

FEConfigTimingDat::~FEConfigTimingDat() {}

void FEConfigTimingDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " (tim_conf_id, logic_id, "
                        "time_par1, time_par2 ) VALUES (:tim_conf_id, :logic_id, :time_par1, :time_par2 )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigTimingDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigTimingDat::writeDB(const EcalLogicID* ecid,
                                const FEConfigTimingDat* item,
                                FEConfigTimingInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigTimingDat::writeDB:  ICONF not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("FEConfigTimingDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getTimingPar1());
    m_writeStmt->setInt(4, item->getTimingPar2());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigTimingDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigTimingDat::fetchData(map<EcalLogicID, FEConfigTimingDat>* fillMap,
                                  FEConfigTimingInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.time_par1, d.time_par2  "
        "FROM channelview cv JOIN " +
        getTable() +
        " d ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE tim_conf_id = :tim_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigTimingDat> p;
    FEConfigTimingDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setTimingPar1(rset->getInt(7));
      dat.setTimingPar2(rset->getInt(8));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigTimingDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigTimingDat::writeArrayDB(const std::map<EcalLogicID, FEConfigTimingDat>* data,
                                     FEConfigTimingInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigTimingDat::writeArrayDB:  tim_conf_id not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iconfid_vec = new int[nrows];
  int* xx = new int[nrows];
  int* yy = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iconf_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigTimingDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, FEConfigTimingDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("FEConfigTimingDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iconfid_vec[count] = iconfID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iconf);
    int x = dataitem->getTimingPar1();
    int y = dataitem->getTimingPar2();

    xx[count] = x;
    yy[count] = y;

    ids_len[count] = sizeof(ids[count]);
    iconf_len[count] = sizeof(iconfid_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]), iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIINT, sizeof(yy[0]), y_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iconfid_vec;
    delete[] xx;
    delete[] yy;

    delete[] ids_len;
    delete[] iconf_len;
    delete[] x_len;
    delete[] y_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigTimingDat::writeArrayDB():  " + e.getMessage()));
  }
}

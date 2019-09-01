#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLinParamDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigLinParamDat::FEConfigLinParamDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_etsat = 0;
}

FEConfigLinParamDat::~FEConfigLinParamDat() {}

void FEConfigLinParamDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " (lin_conf_id, logic_id, "
                        " etsat ) "
                        "VALUES (:lin_conf_id, :logic_id, "
                        ":etsat )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinParamDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigLinParamDat::writeDB(const EcalLogicID* ecid,
                                  const FEConfigLinParamDat* item,
                                  FEConfigLinInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigLinParamDat::writeDB:  ICONF not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("FEConfigLinParamDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getETSat());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinParamDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigLinParamDat::fetchData(map<EcalLogicID, FEConfigLinParamDat>* fillMap,
                                    FEConfigLinInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    //  throw(std::runtime_error("FEConfigLinParamDat::writeDB:  ICONF not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        " d.etsat "
        "FROM channelview cv JOIN " +
        getTable() +
        " d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE lin_conf_id = :lin_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigLinParamDat> p;
    FEConfigLinParamDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setETSat(rset->getFloat(7));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinParamDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigLinParamDat::writeArrayDB(const std::map<EcalLogicID, FEConfigLinParamDat>* data,
                                       FEConfigLinInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigLinParamDat::writeArrayDB:  ICONF not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iov_vec = new int[nrows];
  float* xx = new float[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigLinParamDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, FEConfigLinParamDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("FEConfigLinParamDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iov_vec[count] = iconfID;

    dataitem = &(p->second);
    float x = dataitem->getETSat();

    xx[count] = x;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iov_vec[count]);
    x_len[count] = sizeof(xx[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iov_vec, OCCIINT, sizeof(iov_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT, sizeof(xx[0]), x_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iov_vec;
    delete[] xx;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinParamDat::writeArrayDB():  " + e.getMessage()));
  }
}

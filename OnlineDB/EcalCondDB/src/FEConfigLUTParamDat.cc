#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLUTParamDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigLUTParamDat::FEConfigLUTParamDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_etsat = 0;
  m_tthreshlow = 0;
  m_tthreshhigh = 0;
}

FEConfigLUTParamDat::~FEConfigLUTParamDat() {}

void FEConfigLUTParamDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " (lut_conf_id, logic_id, "
                        " etsat, ttthreshlow, ttthreshhigh ) "
                        "VALUES (:lut_conf_id, :logic_id, "
                        ":etsat, :ttthreshlow, :ttthreshhigh )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLUTParamDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigLUTParamDat::writeDB(const EcalLogicID* ecid,
                                  const FEConfigLUTParamDat* item,
                                  FEConfigLUTInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigLUTParamDat::writeDB:  ICONF not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("FEConfigLUTParamDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getETSat());
    m_writeStmt->setFloat(4, item->getTTThreshlow());
    m_writeStmt->setFloat(5, item->getTTThreshhigh());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLUTParamDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigLUTParamDat::fetchData(map<EcalLogicID, FEConfigLUTParamDat>* fillMap,
                                    FEConfigLUTInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    //  throw(std::runtime_error("FEConfigLUTParamDat::writeDB:  ICONF not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        " d.etsat, d.ttthreshlow, d.ttthreshhigh "
        "FROM channelview cv JOIN " +
        getTable() +
        " d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE lut_conf_id = :lut_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigLUTParamDat> p;
    FEConfigLUTParamDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setETSat(rset->getFloat(7));
      dat.setTTThreshlow(rset->getFloat(8));
      dat.setTTThreshhigh(rset->getFloat(9));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLUTParamDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigLUTParamDat::writeArrayDB(const std::map<EcalLogicID, FEConfigLUTParamDat>* data,
                                       FEConfigLUTInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigLUTParamDat::writeArrayDB:  ICONF not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iov_vec = new int[nrows];
  float* xx = new float[nrows];
  float* yy = new float[nrows];
  float* zz = new float[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* z_len = new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigLUTParamDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, FEConfigLUTParamDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("FEConfigLUTParamDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iov_vec[count] = iconfID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, conf);
    float x = dataitem->getETSat();
    float y = dataitem->getTTThreshlow();
    float z = dataitem->getTTThreshhigh();

    xx[count] = x;
    yy[count] = y;
    zz[count] = z;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iov_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    z_len[count] = sizeof(zz[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iov_vec, OCCIINT, sizeof(iov_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT, sizeof(zz[0]), z_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iov_vec;
    delete[] xx;
    delete[] yy;
    delete[] zz;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;
    delete[] y_len;
    delete[] z_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLUTParamDat::writeArrayDB():  " + e.getMessage()));
  }
}

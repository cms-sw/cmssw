#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCUVFETempDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCUVFETempDat::DCUVFETempDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_vfeTemp = 0;
}

DCUVFETempDat::~DCUVFETempDat() {}

void DCUVFETempDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO dcu_vfe_temp_dat (iov_id, logic_id, "
        "vfe_temp) "
        "VALUES (:iov_id, :logic_id, "
        ":vfe_temp)");
  } catch (SQLException& e) {
    throw(std::runtime_error("DCUVFETempDat::prepareWrite():  " + e.getMessage()));
  }
}

void DCUVFETempDat::writeDB(const EcalLogicID* ecid, const DCUVFETempDat* item, DCUIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("DCUVFETempDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("DCUVFETempDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getVFETemp());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("DCUVFETempDat::writeDB():  " + e.getMessage()));
  }
}

void DCUVFETempDat::fetchData(std::map<EcalLogicID, DCUVFETempDat>* fillMap, DCUIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("DCUVFETempDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.vfe_temp "
        "FROM channelview cv JOIN dcu_vfe_temp_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, DCUVFETempDat> p;
    DCUVFETempDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setVFETemp(rset->getFloat(7));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("DCUVFETempDat::fetchData():  " + e.getMessage()));
  }
}

void DCUVFETempDat::writeArrayDB(const std::map<EcalLogicID, DCUVFETempDat>* data, DCUIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("DCUVFETempDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  float* xx = new float[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];

  const EcalLogicID* channel;
  const DCUVFETempDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, DCUVFETempDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("DCUVFETempDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    float x = dataitem->getVFETemp();

    xx[count] = x;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x_len[count] = sizeof(xx[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT, sizeof(xx[0]), x_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("DCUVFETempDat::writeArrayDB():  " + e.getMessage()));
  }
}

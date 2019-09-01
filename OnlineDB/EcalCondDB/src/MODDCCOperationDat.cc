#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MODDCCOperationDat.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"

using namespace std;
using namespace oracle::occi;

MODDCCOperationDat::MODDCCOperationDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_word = "";
}

MODDCCOperationDat::~MODDCCOperationDat() {}

void MODDCCOperationDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " (iov_id, logic_id, "
                        "operationMode) "
                        "VALUES (:iov_id, :logic_id, "
                        ":1) ");
  } catch (SQLException& e) {
    throw(std::runtime_error("MODDCCOperationDat::prepareWrite():  " + e.getMessage()));
  }
}

void MODDCCOperationDat::writeDB(const EcalLogicID* ecid,
                                 const MODDCCOperationDat* item,
                                 MODRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MODDCCOperationDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MODDCCOperationDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setString(3, item->getOperation());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MODDCCOperationDat::writeDB():  " + e.getMessage()));
  }
}

void MODDCCOperationDat::fetchData(std::map<EcalLogicID, MODDCCOperationDat>* fillMap, MODRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MODDCCOperationDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        " d.operationMode "
        "FROM channelview cv JOIN " +
        getTable() +
        " d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MODDCCOperationDat> p;
    MODDCCOperationDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setOperation(rset->getString(7));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MODDCCOperationDat::fetchData():  " + e.getMessage()));
  }
}

void MODDCCOperationDat::writeArrayDB(const std::map<EcalLogicID, MODDCCOperationDat>* data,
                                      MODRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MODDCCOperationDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  std::string* xx = new std::string[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];

  const EcalLogicID* channel;
  const MODDCCOperationDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, MODDCCOperationDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("MODDCCOperationDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    std::string x = dataitem->getOperation();

    xx[count] = x;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x_len[count] = sizeof(xx[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCISTRING, sizeof(xx[0]), x_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("MonPedestalsDat::writeArrayDB():  " + e.getMessage()));
  }
}

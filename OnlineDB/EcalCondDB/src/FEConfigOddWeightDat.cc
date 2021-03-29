#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigOddWeightDat::FEConfigOddWeightDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_group_id = 0;
}

FEConfigOddWeightDat::~FEConfigOddWeightDat() {}

void FEConfigOddWeightDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO fe_config_weight2_dat (wei2_conf_id, logic_id, "
        "group_id ) "
        "VALUES (:wei2_conf_id, :logic_id, "
        ":group_id )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigOddWeightDat::writeDB(const EcalLogicID* ecid,
                                   const FEConfigOddWeightDat* item,
                                   FEConfigOddWeightInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigOddWeightDat::writeDB:  ICONF not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("FEConfigOddWeightDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getWeightGroupId());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigOddWeightDat::fetchData(map<EcalLogicID, FEConfigOddWeightDat>* fillMap,
                                     FEConfigOddWeightInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    //  throw(std::runtime_error("FEConfigOddWeightDat::writeDB:  ICONF not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.group_id  "
        "FROM channelview cv JOIN fe_config_weight2_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE wei2_conf_id = :wei2_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigOddWeightDat> p;
    FEConfigOddWeightDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setWeightGroupId(rset->getInt(7));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigOddWeightDat::writeArrayDB(const std::map<EcalLogicID, FEConfigOddWeightDat>* data,
                                        FEConfigOddWeightInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigOddWeightDat::writeArrayDB:  ICONF not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iconfid_vec = new int[nrows];
  int* xx = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iconf_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigOddWeightDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, FEConfigOddWeightDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("FEConfigOddWeightDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iconfid_vec[count] = iconfID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iconf);
    int x = dataitem->getWeightGroupId();

    xx[count] = x;

    ids_len[count] = sizeof(ids[count]);
    iconf_len[count] = sizeof(iconfid_vec[count]);

    x_len[count] = sizeof(xx[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]), iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iconfid_vec;
    delete[] xx;

    delete[] ids_len;
    delete[] iconf_len;
    delete[] x_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightDat::writeArrayDB():  " + e.getMessage()));
  }
}

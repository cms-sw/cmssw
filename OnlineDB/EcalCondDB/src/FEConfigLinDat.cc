#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLinDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigLinDat::FEConfigLinDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_multx12 = 0;
  m_multx6 = 0;
  m_multx1 = 0;
  m_shift12 = 0;
  m_shift6 = 0;
  m_shift1 = 0;
}

FEConfigLinDat::~FEConfigLinDat() {}

void FEConfigLinDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO fe_config_lin_dat (lin_conf_id, logic_id, "
        " multx12, multx6, multx1, shift12, shift6, shift1 ) "
        "VALUES (:lin_conf_id, :logic_id, "
        ":multx12, :multx6, :multx1, :shift12, :shift6, :shift1 )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigLinDat::writeDB(const EcalLogicID* ecid,
                             const FEConfigLinDat* item,
                             FEConfigLinInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigLinDat::writeDB:  ICONF not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("FEConfigLinDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getMultX12());
    m_writeStmt->setInt(4, item->getMultX6());
    m_writeStmt->setInt(5, item->getMultX1());
    m_writeStmt->setInt(6, item->getShift12());
    m_writeStmt->setInt(7, item->getShift6());
    m_writeStmt->setInt(8, item->getShift1());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigLinDat::fetchData(map<EcalLogicID, FEConfigLinDat>* fillMap, FEConfigLinInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    //  throw(std::runtime_error("FEConfigLinDat::writeDB:  ICONF not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.multx12, d.multx6, d.multx1, d.shift12, d.shift6, d.shift1 "
        "FROM channelview cv JOIN fe_config_lin_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE lin_conf_id = :lin_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigLinDat> p;
    FEConfigLinDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setMultX12(rset->getInt(7));
      dat.setMultX6(rset->getInt(8));
      dat.setMultX1(rset->getInt(9));
      dat.setShift12(rset->getInt(10));
      dat.setShift6(rset->getInt(11));
      dat.setShift1(rset->getInt(12));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigLinDat::writeArrayDB(const std::map<EcalLogicID, FEConfigLinDat>* data,
                                  FEConfigLinInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigLinDat::writeArrayDB:  ICONF not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iconfid_vec = new int[nrows];
  int* xx = new int[nrows];
  int* yy = new int[nrows];
  int* zz = new int[nrows];
  int* ww = new int[nrows];
  int* rr = new int[nrows];
  int* ss = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iconf_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* z_len = new ub2[nrows];
  ub2* w_len = new ub2[nrows];
  ub2* r_len = new ub2[nrows];
  ub2* s_len = new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigLinDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, FEConfigLinDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("FEConfigLinDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iconfid_vec[count] = iconfID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iconf);
    int x = dataitem->getMultX12();
    int y = dataitem->getMultX6();
    int z = dataitem->getMultX1();
    int w = dataitem->getShift12();
    int r = dataitem->getShift6();
    int s = dataitem->getShift1();

    xx[count] = x;
    yy[count] = y;
    zz[count] = z;
    ww[count] = w;
    rr[count] = r;
    ss[count] = s;

    ids_len[count] = sizeof(ids[count]);
    iconf_len[count] = sizeof(iconfid_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    z_len[count] = sizeof(zz[count]);
    w_len[count] = sizeof(ww[count]);
    r_len[count] = sizeof(rr[count]);
    s_len[count] = sizeof(ss[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]), iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIINT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIINT, sizeof(zz[0]), z_len);
    m_writeStmt->setDataBuffer(6, (dvoid*)ww, OCCIINT, sizeof(ww[0]), w_len);
    m_writeStmt->setDataBuffer(7, (dvoid*)rr, OCCIINT, sizeof(rr[0]), r_len);
    m_writeStmt->setDataBuffer(8, (dvoid*)ss, OCCIINT, sizeof(ss[0]), s_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iconfid_vec;
    delete[] xx;
    delete[] yy;
    delete[] zz;
    delete[] ww;
    delete[] rr;
    delete[] ss;

    delete[] ids_len;
    delete[] iconf_len;
    delete[] x_len;
    delete[] y_len;
    delete[] z_len;
    delete[] w_len;
    delete[] r_len;
    delete[] s_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigLinDat::writeArrayDB():  " + e.getMessage()));
  }
}

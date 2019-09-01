#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCULVRVoltagesDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCULVRVoltagesDat::DCULVRVoltagesDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_vfe1_A = 0;
  m_vfe2_A = 0;
  m_vfe3_A = 0;
  m_vfe4_A = 0;
  m_vfe5_A = 0;
  m_VCC = 0;
  m_vfe4_5_D = 0;
  m_vfe1_2_3_D = 0;
  m_buffer = 0;
  m_fenix = 0;
  m_V43_A = 0;
  m_OCM = 0;
  m_GOH = 0;
  m_INH = 0;
  m_V43_D = 0;
}

DCULVRVoltagesDat::~DCULVRVoltagesDat() {}

void DCULVRVoltagesDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO dcu_lvr_voltages_dat (iov_id, logic_id, "
        "vfe1_A, vfe2_A, vfe3_A, vfe4_A, vfe5_A, VCC, vfe4_5_D, vfe1_2_3_D, buffer, fenix, V43_A, OCM, GOH, INH, "
        "V43_D) "
        "VALUES (:iov_id, :logic_id, "
        ":3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17)");
  } catch (SQLException& e) {
    throw(std::runtime_error("DCULVRVoltagesDat::prepareWrite():  " + e.getMessage()));
  }
}

void DCULVRVoltagesDat::writeDB(const EcalLogicID* ecid, const DCULVRVoltagesDat* item, DCUIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("DCULVRVoltagesDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("DCULVRVoltagesDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getVFE1_A());
    m_writeStmt->setFloat(4, item->getVFE2_A());
    m_writeStmt->setFloat(5, item->getVFE3_A());
    m_writeStmt->setFloat(6, item->getVFE4_A());
    m_writeStmt->setFloat(7, item->getVFE5_A());
    m_writeStmt->setFloat(8, item->getVCC());
    m_writeStmt->setFloat(9, item->getVFE4_5_D());
    m_writeStmt->setFloat(10, item->getVFE1_2_3_D());
    m_writeStmt->setFloat(11, item->getBuffer());
    m_writeStmt->setFloat(12, item->getFenix());
    m_writeStmt->setFloat(13, item->getV43_A());
    m_writeStmt->setFloat(14, item->getOCM());
    m_writeStmt->setFloat(15, item->getGOH());
    m_writeStmt->setFloat(16, item->getINH());
    m_writeStmt->setFloat(17, item->getV43_D());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("DCULVRVoltagesDat::writeDB():  " + e.getMessage()));
  }
}

void DCULVRVoltagesDat::fetchData(std::map<EcalLogicID, DCULVRVoltagesDat>* fillMap, DCUIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("DCULVRVoltagesDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.vfe1_A, d.vfe2_A, d.vfe3_A, d.vfe4_A, d.vfe5_A, d.VCC, d.vfe4_5_D, d.vfe1_2_3_D, d.buffer, d.fenix, "
        "d.V43_A, d.OCM, d.GOH, d.INH, d.V43_D  "
        "FROM channelview cv JOIN dcu_lvr_voltages_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, DCULVRVoltagesDat> p;
    DCULVRVoltagesDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setVFE1_A(rset->getFloat(7));
      dat.setVFE2_A(rset->getFloat(8));
      dat.setVFE3_A(rset->getFloat(9));
      dat.setVFE4_A(rset->getFloat(10));
      dat.setVFE5_A(rset->getFloat(11));
      dat.setVCC(rset->getFloat(12));
      dat.setVFE4_5_D(rset->getFloat(13));
      dat.setVFE1_2_3_D(rset->getFloat(14));
      dat.setBuffer(rset->getFloat(15));
      dat.setFenix(rset->getFloat(16));
      dat.setV43_A(rset->getFloat(17));
      dat.setOCM(rset->getFloat(18));
      dat.setGOH(rset->getFloat(19));
      dat.setINH(rset->getFloat(20));
      dat.setV43_D(rset->getFloat(21));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("DCULVRVoltagesDat::fetchData():  " + e.getMessage()));
  }
}
void DCULVRVoltagesDat::writeArrayDB(const std::map<EcalLogicID, DCULVRVoltagesDat>* data,
                                     DCUIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("DCULVRVoltagesDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  float* xx = new float[nrows];
  float* yy = new float[nrows];
  float* zz = new float[nrows];
  float* ww = new float[nrows];
  float* uu = new float[nrows];
  float* tt = new float[nrows];
  float* rr = new float[nrows];
  float* pp = new float[nrows];
  float* ll = new float[nrows];
  float* mm = new float[nrows];
  float* nn = new float[nrows];
  float* qq = new float[nrows];
  float* ss = new float[nrows];
  float* vv = new float[nrows];
  float* hh = new float[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* z_len = new ub2[nrows];
  ub2* w_len = new ub2[nrows];
  ub2* u_len = new ub2[nrows];
  ub2* t_len = new ub2[nrows];
  ub2* r_len = new ub2[nrows];
  ub2* p_len = new ub2[nrows];
  ub2* l_len = new ub2[nrows];
  ub2* m_len = new ub2[nrows];
  ub2* n_len = new ub2[nrows];
  ub2* q_len = new ub2[nrows];
  ub2* s_len = new ub2[nrows];
  ub2* v_len = new ub2[nrows];
  ub2* h_len = new ub2[nrows];

  const EcalLogicID* channel;
  const DCULVRVoltagesDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, DCULVRVoltagesDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("DCULVRVoltagesDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    float x = dataitem->getVFE1_A();
    float y = dataitem->getVFE2_A();
    float z = dataitem->getVFE3_A();
    float w = dataitem->getVFE4_A();
    float u = dataitem->getVFE5_A();
    float t = dataitem->getVCC();
    float r = dataitem->getVFE4_5_D();
    float pi = dataitem->getVFE1_2_3_D();
    float l = dataitem->getBuffer();
    float m = dataitem->getFenix();
    float n = dataitem->getV43_A();
    float q = dataitem->getOCM();
    float s = dataitem->getGOH();
    float v = dataitem->getINH();
    float h = dataitem->getV43_D();

    xx[count] = x;
    yy[count] = y;
    zz[count] = z;
    ww[count] = w;
    uu[count] = u;
    tt[count] = t;
    rr[count] = r;
    pp[count] = pi;
    ll[count] = l;
    mm[count] = m;
    nn[count] = n;
    qq[count] = q;
    ss[count] = s;
    vv[count] = v;
    hh[count] = h;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    z_len[count] = sizeof(zz[count]);
    w_len[count] = sizeof(ww[count]);
    u_len[count] = sizeof(uu[count]);
    t_len[count] = sizeof(tt[count]);
    r_len[count] = sizeof(rr[count]);
    p_len[count] = sizeof(pp[count]);
    l_len[count] = sizeof(ll[count]);
    m_len[count] = sizeof(mm[count]);
    n_len[count] = sizeof(nn[count]);
    q_len[count] = sizeof(qq[count]);
    s_len[count] = sizeof(ss[count]);
    v_len[count] = sizeof(vv[count]);
    h_len[count] = sizeof(hh[count]);
    count++;
  }
  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT, sizeof(zz[0]), z_len);
    m_writeStmt->setDataBuffer(6, (dvoid*)ww, OCCIFLOAT, sizeof(ww[0]), w_len);
    m_writeStmt->setDataBuffer(7, (dvoid*)uu, OCCIFLOAT, sizeof(uu[0]), u_len);
    m_writeStmt->setDataBuffer(8, (dvoid*)tt, OCCIFLOAT, sizeof(tt[0]), t_len);
    m_writeStmt->setDataBuffer(9, (dvoid*)rr, OCCIFLOAT, sizeof(rr[0]), r_len);
    m_writeStmt->setDataBuffer(10, (dvoid*)pp, OCCIFLOAT, sizeof(pp[0]), p_len);
    m_writeStmt->setDataBuffer(11, (dvoid*)ll, OCCIFLOAT, sizeof(ll[0]), l_len);
    m_writeStmt->setDataBuffer(12, (dvoid*)mm, OCCIFLOAT, sizeof(mm[0]), m_len);
    m_writeStmt->setDataBuffer(13, (dvoid*)nn, OCCIFLOAT, sizeof(nn[0]), n_len);
    m_writeStmt->setDataBuffer(14, (dvoid*)qq, OCCIFLOAT, sizeof(qq[0]), q_len);
    m_writeStmt->setDataBuffer(15, (dvoid*)ss, OCCIFLOAT, sizeof(ss[0]), s_len);
    m_writeStmt->setDataBuffer(16, (dvoid*)vv, OCCIFLOAT, sizeof(vv[0]), v_len);
    m_writeStmt->setDataBuffer(17, (dvoid*)hh, OCCIFLOAT, sizeof(hh[0]), h_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx;
    delete[] yy;
    delete[] zz;
    delete[] ww;
    delete[] uu;
    delete[] tt;
    delete[] rr;
    delete[] pp;
    delete[] ll;
    delete[] mm;
    delete[] nn;
    delete[] qq;
    delete[] ss;
    delete[] vv;
    delete[] hh;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;
    delete[] y_len;
    delete[] z_len;
    delete[] w_len;
    delete[] u_len;
    delete[] t_len;
    delete[] r_len;
    delete[] p_len;
    delete[] l_len;
    delete[] m_len;
    delete[] n_len;
    delete[] q_len;
    delete[] s_len;
    delete[] v_len;
    delete[] h_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("DCULVRVoltagesDat::writeArrayDB():  " + e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserPulseDat.h"

using namespace std;
using namespace oracle::occi;

MonLaserPulseDat::MonLaserPulseDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_pulseHeightMean = 0;
  m_pulseHeightRMS = 0;
  m_pulseWidthMean = 0;
  m_pulseWidthRMS = 0;
}

MonLaserPulseDat::~MonLaserPulseDat() {}

void MonLaserPulseDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO mon_laser_pulse_dat (iov_id, logic_id, "
        "pulse_height_mean, pulse_height_rms, pulse_width_mean, pulse_width_rms) "
        "VALUES (:iov_id, :logic_id, "
        ":3, :4, :5, :6)");
  } catch (SQLException& e) {
    throw(std::runtime_error("MonLaserPulseDat::prepareWrite():  " + e.getMessage()));
  }
}

void MonLaserPulseDat::writeDB(const EcalLogicID* ecid, const MonLaserPulseDat* item, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonLaserPulseDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MonLaserPulseDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getPulseHeightMean());
    m_writeStmt->setFloat(4, item->getPulseHeightRMS());
    m_writeStmt->setFloat(5, item->getPulseWidthMean());
    m_writeStmt->setFloat(6, item->getPulseWidthRMS());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MonLaserPulseDat::writeDB():  " + e.getMessage()));
  }
}

void MonLaserPulseDat::fetchData(std::map<EcalLogicID, MonLaserPulseDat>* fillMap, MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MonLaserPulseDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        "d.pulse_height_mean, d.pulse_height_rms, d.pulse_width_mean, d.pulse_width_rms "
        "FROM channelview cv JOIN mon_laser_pulse_dat d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MonLaserPulseDat> p;
    MonLaserPulseDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setPulseHeightMean(rset->getFloat(7));
      dat.setPulseHeightRMS(rset->getFloat(8));
      dat.setPulseWidthMean(rset->getFloat(9));
      dat.setPulseWidthRMS(rset->getFloat(10));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MonLaserPulseDat::fetchData():  " + e.getMessage()));
  }
}

void MonLaserPulseDat::writeArrayDB(const std::map<EcalLogicID, MonLaserPulseDat>* data,
                                    MonRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MonLaserPulseDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  float* xx = new float[nrows];
  float* yy = new float[nrows];
  float* zz = new float[nrows];
  float* ww = new float[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* z_len = new ub2[nrows];
  ub2* w_len = new ub2[nrows];

  const EcalLogicID* channel;
  const MonLaserPulseDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, MonLaserPulseDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("MonLaserPulseDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    float x = dataitem->getPulseHeightMean();
    float y = dataitem->getPulseHeightRMS();
    float z = dataitem->getPulseWidthMean();
    float w = dataitem->getPulseWidthRMS();

    xx[count] = x;
    yy[count] = y;
    zz[count] = z;
    ww[count] = w;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    z_len[count] = sizeof(zz[count]);
    w_len[count] = sizeof(ww[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT, sizeof(zz[0]), z_len);
    m_writeStmt->setDataBuffer(6, (dvoid*)ww, OCCIFLOAT, sizeof(ww[0]), w_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx;
    delete[] yy;
    delete[] zz;
    delete[] ww;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x_len;
    delete[] y_len;
    delete[] z_len;
    delete[] w_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("MonLaserPulseDat::writeArrayDB():  " + e.getMessage()));
  }
}

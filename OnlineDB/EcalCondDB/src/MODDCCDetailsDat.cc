#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MODDCCDetailsDat.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"

using namespace std;
using namespace oracle::occi;

MODDCCDetailsDat::MODDCCDetailsDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_qpll = 0;
  m_opto = 0;
  m_tout = 0;
  m_head = 0;
  m_evnu = 0;
  m_bxnu = 0;
  m_evpa = 0;
  m_odpa = 0;
  m_blsi = 0;
  m_alff = 0;
  m_fuff = 0;
  m_fusu = 0;
}

MODDCCDetailsDat::~MODDCCDetailsDat() {}

void MODDCCDetailsDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(" INSERT INTO " + getTable() +
                        " (iov_id, logic_id, "
                        " qpll_error, optical_link, data_timeout, dcc_header, event_number, bx_number, "
                        " even_parity, odd_parity, block_size, almost_full_fifo, full_fifo, "
                        " forced_full_supp ) "
                        " VALUES (:iov_id, :logic_id, "
                        " :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12 ) ");
  } catch (SQLException& e) {
    throw(std::runtime_error("MODDCCDetailsDat::prepareWrite():  " + e.getMessage()));
  }
}

void MODDCCDetailsDat::writeDB(const EcalLogicID* ecid, const MODDCCDetailsDat* item, MODRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MODDCCDetailsDat::writeDB:  IOV not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("MODDCCDetailsDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getQPLL());
    m_writeStmt->setInt(4, item->getOpticalLink());
    m_writeStmt->setInt(5, item->getDataTimeout());
    m_writeStmt->setInt(6, item->getHeader());
    m_writeStmt->setInt(7, item->getEventNumber());
    m_writeStmt->setInt(8, item->getBXNumber());
    m_writeStmt->setInt(9, item->getEvenParity());
    m_writeStmt->setInt(10, item->getOddParity());
    m_writeStmt->setInt(11, item->getBlockSize());
    m_writeStmt->setInt(12, item->getAlmostFullFIFO());
    m_writeStmt->setInt(13, item->getFullFIFO());
    m_writeStmt->setInt(14, item->getForcedFullSupp());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("MODDCCDetailsDat::writeDB():  " + e.getMessage()));
  }
}

void MODDCCDetailsDat::fetchData(std::map<EcalLogicID, MODDCCDetailsDat>* fillMap, MODRunIOV* iov) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    //  throw(std::runtime_error("MODDCCDetailsDat::writeDB:  IOV not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        " d.qpll_error, d.optical_link, d.data_timeout, d.dcc_header, d.event_number, d.bx_number, d.even_parity, "
        "d.odd_parity, d.block_size, d.almost_full_fifo, d.full_fifo, d.forced_full_supp "
        "FROM channelview cv JOIN " +
        getTable() +
        " d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, MODDCCDetailsDat> p;
    MODDCCDetailsDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setQPLL(rset->getInt(7));
      dat.setOpticalLink(rset->getInt(8));
      dat.setDataTimeout(rset->getInt(9));
      dat.setHeader(rset->getInt(10));
      dat.setEventNumber(rset->getInt(11));
      dat.setBXNumber(rset->getInt(12));
      dat.setEvenParity(rset->getInt(13));
      dat.setOddParity(rset->getInt(14));
      dat.setBlockSize(rset->getInt(15));
      dat.setAlmostFullFIFO(rset->getInt(16));
      dat.setFullFIFO(rset->getInt(17));
      dat.setForcedFullSupp(rset->getInt(18));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MODDCCDetailsDat::fetchData():  " + e.getMessage()));
  }
}

void MODDCCDetailsDat::writeArrayDB(const std::map<EcalLogicID, MODDCCDetailsDat>* data,
                                    MODRunIOV* iov) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("MODDCCDetailsDat::writeArrayDB:  IOV not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iovid_vec = new int[nrows];
  int* xx1 = new int[nrows];
  int* xx2 = new int[nrows];
  int* xx3 = new int[nrows];
  int* xx4 = new int[nrows];
  int* xx5 = new int[nrows];
  int* xx6 = new int[nrows];
  int* xx7 = new int[nrows];
  int* xx8 = new int[nrows];
  int* xx9 = new int[nrows];
  int* xx10 = new int[nrows];
  int* xx11 = new int[nrows];
  int* xx12 = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iov_len = new ub2[nrows];
  ub2* x1_len = new ub2[nrows];
  ub2* x2_len = new ub2[nrows];
  ub2* x3_len = new ub2[nrows];
  ub2* x4_len = new ub2[nrows];
  ub2* x5_len = new ub2[nrows];
  ub2* x6_len = new ub2[nrows];
  ub2* x7_len = new ub2[nrows];
  ub2* x8_len = new ub2[nrows];
  ub2* x9_len = new ub2[nrows];
  ub2* x10_len = new ub2[nrows];
  ub2* x11_len = new ub2[nrows];
  ub2* x12_len = new ub2[nrows];

  const EcalLogicID* channel;
  const MODDCCDetailsDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, MODDCCDetailsDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("MODDCCDetailsDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iovid_vec[count] = iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    int x1 = dataitem->getQPLL();
    int x2 = dataitem->getOpticalLink();
    int x3 = dataitem->getDataTimeout();
    int x4 = dataitem->getHeader();
    int x5 = dataitem->getEventNumber();
    int x6 = dataitem->getBXNumber();
    int x7 = dataitem->getEvenParity();
    int x8 = dataitem->getOddParity();
    int x9 = dataitem->getBlockSize();
    int x10 = dataitem->getAlmostFullFIFO();
    int x11 = dataitem->getFullFIFO();
    int x12 = dataitem->getForcedFullSupp();

    xx1[count] = x1;
    xx2[count] = x2;
    xx3[count] = x3;
    xx4[count] = x4;
    xx5[count] = x5;
    xx6[count] = x6;
    xx7[count] = x7;
    xx8[count] = x8;
    xx9[count] = x9;
    xx10[count] = x10;
    xx11[count] = x11;
    xx12[count] = x12;

    ids_len[count] = sizeof(ids[count]);
    iov_len[count] = sizeof(iovid_vec[count]);

    x1_len[count] = sizeof(xx1[count]);
    x2_len[count] = sizeof(xx2[count]);
    x3_len[count] = sizeof(xx3[count]);
    x4_len[count] = sizeof(xx4[count]);
    x5_len[count] = sizeof(xx5[count]);
    x6_len[count] = sizeof(xx6[count]);
    x7_len[count] = sizeof(xx7[count]);
    x8_len[count] = sizeof(xx8[count]);
    x9_len[count] = sizeof(xx9[count]);
    x10_len[count] = sizeof(xx10[count]);
    x11_len[count] = sizeof(xx11[count]);
    x12_len[count] = sizeof(xx12[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]), iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)xx1, OCCIINT, sizeof(xx1[0]), x1_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)xx2, OCCIINT, sizeof(xx2[0]), x2_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)xx3, OCCIINT, sizeof(xx3[0]), x3_len);
    m_writeStmt->setDataBuffer(6, (dvoid*)xx4, OCCIINT, sizeof(xx4[0]), x4_len);
    m_writeStmt->setDataBuffer(7, (dvoid*)xx5, OCCIINT, sizeof(xx5[0]), x5_len);
    m_writeStmt->setDataBuffer(8, (dvoid*)xx6, OCCIINT, sizeof(xx6[0]), x6_len);
    m_writeStmt->setDataBuffer(9, (dvoid*)xx7, OCCIINT, sizeof(xx7[0]), x7_len);
    m_writeStmt->setDataBuffer(10, (dvoid*)xx8, OCCIINT, sizeof(xx8[0]), x8_len);
    m_writeStmt->setDataBuffer(11, (dvoid*)xx9, OCCIINT, sizeof(xx9[0]), x9_len);
    m_writeStmt->setDataBuffer(12, (dvoid*)xx10, OCCIINT, sizeof(xx10[0]), x10_len);
    m_writeStmt->setDataBuffer(13, (dvoid*)xx11, OCCIINT, sizeof(xx11[0]), x11_len);
    m_writeStmt->setDataBuffer(14, (dvoid*)xx12, OCCIINT, sizeof(xx12[0]), x12_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iovid_vec;
    delete[] xx1;
    delete[] xx2;
    delete[] xx3;
    delete[] xx4;
    delete[] xx5;
    delete[] xx6;
    delete[] xx7;
    delete[] xx8;
    delete[] xx9;
    delete[] xx10;
    delete[] xx11;
    delete[] xx12;

    delete[] ids_len;
    delete[] iov_len;
    delete[] x1_len;
    delete[] x2_len;
    delete[] x3_len;
    delete[] x4_len;
    delete[] x5_len;
    delete[] x6_len;
    delete[] x7_len;
    delete[] x8_len;
    delete[] x9_len;
    delete[] x10_len;
    delete[] x11_len;
    delete[] x12_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("MonPedestalsDat::writeArrayDB():  " + e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODPedestalOffsetsDat.h"

using namespace std;
using namespace oracle::occi;

ODPedestalOffsetsDat::ODPedestalOffsetsDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_sm = 0;
  m_fed = 0;
  m_tt = 0;
  m_xt = 0;
  m_low = 0;
  m_mid = 0;
  m_high = 0;
}

ODPedestalOffsetsDat::~ODPedestalOffsetsDat() {}

void ODPedestalOffsetsDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " (rec_id, sm_id, fed_id, tt_id, cry_id, low, mid, high) "
                        "VALUES (:1, :2, :3, :4, :5, :6, :7, :8 )");
  } catch (SQLException& e) {
    throw(std::runtime_error("ODPedestalOffsetsDat::prepareWrite():  " + e.getMessage()));
  }
}

void ODPedestalOffsetsDat::writeDB(const ODPedestalOffsetsDat* item, ODFEPedestalOffsetInfo* iov) noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getSMId());
    m_writeStmt->setInt(3, item->getFedId());
    m_writeStmt->setInt(4, item->getTTId());
    m_writeStmt->setInt(5, item->getCrystalId());
    m_writeStmt->setInt(6, item->getLow());
    m_writeStmt->setInt(7, item->getMid());
    m_writeStmt->setInt(8, item->getHigh());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("ODPedestalOffsetsDat::writeDB():  " + e.getMessage()));
  }
}

void ODPedestalOffsetsDat::fetchData(std::vector<ODPedestalOffsetsDat>* p,
                                     ODFEPedestalOffsetInfo* iov) noexcept(false) {
  this->checkConnection();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) {
    std::cout << "ID not in the DB" << endl;
    return;
  }

  try {
    m_readStmt->setSQL("SELECT * FROM " + getTable() + " WHERE rec_id = :rec_id order by sm_id, fed_id, tt_id, cry_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();

    //    std::vector< ODPedestalOffsetsDat > p;
    ODPedestalOffsetsDat dat;
    while (rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setSMId(rset->getInt(2));
      dat.setFedId(rset->getInt(3));
      dat.setTTId(rset->getInt(4));
      dat.setCrystalId(rset->getInt(5));
      dat.setLow(rset->getInt(6));
      dat.setMid(rset->getInt(7));
      dat.setHigh(rset->getInt(8));

      p->push_back(dat);
    }

  } catch (SQLException& e) {
    throw(std::runtime_error("ODPedestalOffsetsDat::fetchData():  " + e.getMessage()));
  }
}

//  ************************************************************************   //

void ODPedestalOffsetsDat::writeArrayDB(const std::vector<ODPedestalOffsetsDat>& data,
                                        ODFEPedestalOffsetInfo* iov) noexcept(false) {
  this->checkConnection();

  int iovID = iov->fetchID();
  if (!iovID) {
    throw(std::runtime_error("ODDelays::writeArrayDB:  ODFEDelaysInfo not in DB"));
  }

  int nrows = data.size();
  int* ids = new int[nrows];
  int* xx = new int[nrows];
  int* yy = new int[nrows];
  int* zz = new int[nrows];
  int* st = new int[nrows];
  int* xx1 = new int[nrows];
  int* yy1 = new int[nrows];
  int* zz1 = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* x_len = new ub2[nrows];
  ub2* y_len = new ub2[nrows];
  ub2* z_len = new ub2[nrows];
  ub2* st_len = new ub2[nrows];
  ub2* x1_len = new ub2[nrows];
  ub2* y1_len = new ub2[nrows];
  ub2* z1_len = new ub2[nrows];

  ODPedestalOffsetsDat dataitem;

  int n_data = (int)data.size();
  for (int count = 0; count < n_data; count++) {
    dataitem = data[count];
    ids[count] = iovID;
    xx[count] = dataitem.getSMId();
    yy[count] = dataitem.getFedId();
    zz[count] = dataitem.getTTId();
    st[count] = dataitem.getCrystalId();
    xx1[count] = dataitem.getLow();
    yy1[count] = dataitem.getMid();
    zz1[count] = dataitem.getHigh();

    ids_len[count] = sizeof(ids[count]);
    x_len[count] = sizeof(xx[count]);
    y_len[count] = sizeof(yy[count]);
    z_len[count] = sizeof(zz[count]);
    st_len[count] = sizeof(st[count]);
    x1_len[count] = sizeof(xx1[count]);
    y1_len[count] = sizeof(yy1[count]);
    z1_len[count] = sizeof(zz1[count]);
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len);
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT, sizeof(yy[0]), y_len);
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT, sizeof(zz[0]), z_len);
    m_writeStmt->setDataBuffer(5, (dvoid*)st, OCCIINT, sizeof(st[0]), st_len);
    m_writeStmt->setDataBuffer(6, (dvoid*)xx1, OCCIINT, sizeof(xx1[0]), x1_len);
    m_writeStmt->setDataBuffer(7, (dvoid*)yy1, OCCIINT, sizeof(yy1[0]), y1_len);
    m_writeStmt->setDataBuffer(8, (dvoid*)zz1, OCCIINT, sizeof(zz1[0]), z1_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] xx;
    delete[] yy;
    delete[] zz;
    delete[] st;
    delete[] xx1;
    delete[] yy1;
    delete[] zz1;

    delete[] ids_len;
    delete[] x_len;
    delete[] y_len;
    delete[] z_len;
    delete[] st_len;
    delete[] x1_len;
    delete[] y1_len;
    delete[] z1_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("ODPedestalOffsetsDat::writeArrayDB():  " + e.getMessage()));
  }
}

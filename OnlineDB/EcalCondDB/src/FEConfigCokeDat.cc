#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigCokeDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigCokeInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigCokeDat::FEConfigCokeDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  clear();
}

FEConfigCokeDat::~FEConfigCokeDat() {}

void FEConfigCokeDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO " + getTable() +
        " (coke_conf_id, logic_id, "
        " THRESHOLD, SUCC_EVENT_LIMIT, CUMUL_EVENT_LIMIT, SUCC_DETECT_ENABLE, CUMUL_DETECT_ENABLE, THD1_THRESHOLD, "
        "SUCC1_EV_LIMIT, CUMUL1_EV_LIMIT, COMBI_MODE, OCC_MODE, COMB_SUCC_DETECT, COMB_CUMUL_DETECT, OCC_DETECT, "
        "CUMUL1_DETECT, THD2_THRESHOLD , OCC_LIMIT , THD3_THRESHOLD , CUMUL2_LIMIT , STOP_BUFW  ) "
        "VALUES (:coke_conf_id, :logic_id, "
        ":m1, :m2, :m3, :m4, :m5, :m6, :m7, :m8, :m9, :m10, :m11, :m12, :m13, :m14, :m15, :m16, :m17, :m18, :m19 )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigCokeDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigCokeDat::writeDB(const EcalLogicID* ecid,
                              const FEConfigCokeDat* item,
                              FEConfigCokeInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigCokeDat::writeDB:  ICONF not in DB"));
  }

  int logicID = ecid->getLogicID();
  if (!logicID) {
    throw(std::runtime_error("FEConfigCokeDat::writeDB:  Bad EcalLogicID"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getPar1());
    m_writeStmt->setInt(4, item->getPar2());
    m_writeStmt->setInt(5, item->getPar3());
    m_writeStmt->setInt(6, item->getPar4());
    m_writeStmt->setInt(7, item->getPar5());
    m_writeStmt->setInt(8, item->getPar6());
    m_writeStmt->setInt(9, item->getPar7());
    m_writeStmt->setInt(10, item->getPar8());
    m_writeStmt->setInt(11, item->getPar9());
    m_writeStmt->setInt(12, item->getPar10());
    m_writeStmt->setInt(13, item->getPar11());
    m_writeStmt->setInt(14, item->getPar12());
    m_writeStmt->setInt(15, item->getPar13());
    m_writeStmt->setInt(16, item->getPar14());
    m_writeStmt->setInt(17, item->getPar15());
    m_writeStmt->setInt(18, item->getPar16());
    m_writeStmt->setInt(19, item->getPar17());
    m_writeStmt->setInt(20, item->getPar18());
    m_writeStmt->setInt(21, item->getPar19());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigCokeDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigCokeDat::fetchData(map<EcalLogicID, FEConfigCokeDat>* fillMap, FEConfigCokeInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    //  throw(std::runtime_error("FEConfigCokeDat::writeDB:  ICONF not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
        " d.THRESHOLD, d.SUCC_EVENT_LIMIT, d.CUMUL_EVENT_LIMIT, d.SUCC_DETECT_ENABLE, d.CUMUL_DETECT_ENABLE, "
        "d.THD1_THRESHOLD, d.SUCC1_EV_LIMIT, d.CUMUL1_EV_LIMIT, d.COMBI_MODE, d.OCC_MODE, d.COMB_SUCC_DETECT, "
        "d.COMB_CUMUL_DETECT, d.OCC_DETECT, d.CUMUL1_DETECT, d.THD2_THRESHOLD , d.OCC_LIMIT , d.THD3_THRESHOLD , "
        "d.CUMUL2_LIMIT , d.STOP_BUFW  "
        "FROM channelview cv JOIN " +
        getTable() +
        " d "
        "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
        "WHERE coke_conf_id = :coke_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigCokeDat> p;
    FEConfigCokeDat dat;
    while (rset->next()) {
      p.first = EcalLogicID(rset->getString(1),   // name
                            rset->getInt(2),      // logic_id
                            rset->getInt(3),      // id1
                            rset->getInt(4),      // id2
                            rset->getInt(5),      // id3
                            rset->getString(6));  // maps_to

      dat.setPar1(rset->getInt(7));
      dat.setPar2(rset->getInt(8));
      dat.setPar3(rset->getInt(9));
      dat.setPar4(rset->getInt(10));
      dat.setPar5(rset->getInt(11));
      dat.setPar6(rset->getInt(12));
      dat.setPar7(rset->getInt(13));
      dat.setPar8(rset->getInt(14));
      dat.setPar9(rset->getInt(15));
      dat.setPar10(rset->getInt(16));
      dat.setPar11(rset->getInt(17));
      dat.setPar12(rset->getInt(18));
      dat.setPar13(rset->getInt(19));
      dat.setPar14(rset->getInt(20));
      dat.setPar15(rset->getInt(21));
      dat.setPar16(rset->getInt(22));
      dat.setPar17(rset->getInt(23));
      dat.setPar18(rset->getInt(24));
      dat.setPar19(rset->getInt(25));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigCokeDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigCokeDat::writeArrayDB(const std::map<EcalLogicID, FEConfigCokeDat>* data,
                                   FEConfigCokeInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigCokeDat::writeArrayDB:  ICONF not in DB"));
  }

  int nrows = data->size();
  int* ids = new int[nrows];
  int* iconfid_vec = new int[nrows];
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
  int* xx13 = new int[nrows];
  int* xx14 = new int[nrows];
  int* xx15 = new int[nrows];
  int* xx16 = new int[nrows];
  int* xx17 = new int[nrows];
  int* xx18 = new int[nrows];
  int* xx19 = new int[nrows];

  ub2* ids_len = new ub2[nrows];
  ub2* iconf_len = new ub2[nrows];
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
  ub2* x13_len = new ub2[nrows];
  ub2* x14_len = new ub2[nrows];
  ub2* x15_len = new ub2[nrows];
  ub2* x16_len = new ub2[nrows];
  ub2* x17_len = new ub2[nrows];
  ub2* x18_len = new ub2[nrows];
  ub2* x19_len = new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigCokeDat* dataitem;
  int count = 0;
  typedef map<EcalLogicID, FEConfigCokeDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) {
      throw(std::runtime_error("FEConfigCokeDat::writeArrayDB:  Bad EcalLogicID"));
    }
    ids[count] = logicID;
    iconfid_vec[count] = iconfID;

    dataitem = &(p->second);

    xx1[count] = dataitem->getPar1();
    xx2[count] = dataitem->getPar2();
    xx3[count] = dataitem->getPar3();
    xx4[count] = dataitem->getPar4();
    xx5[count] = dataitem->getPar5();
    xx6[count] = dataitem->getPar6();
    xx7[count] = dataitem->getPar7();
    xx8[count] = dataitem->getPar8();
    xx9[count] = dataitem->getPar9();
    xx10[count] = dataitem->getPar10();
    xx11[count] = dataitem->getPar11();
    xx12[count] = dataitem->getPar12();
    xx13[count] = dataitem->getPar13();
    xx14[count] = dataitem->getPar14();
    xx15[count] = dataitem->getPar15();
    xx16[count] = dataitem->getPar16();
    xx17[count] = dataitem->getPar17();
    xx18[count] = dataitem->getPar18();
    xx19[count] = dataitem->getPar19();

    ids_len[count] = sizeof(ids[count]);
    iconf_len[count] = sizeof(iconfid_vec[count]);

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
    x13_len[count] = sizeof(xx13[count]);
    x14_len[count] = sizeof(xx14[count]);
    x15_len[count] = sizeof(xx15[count]);
    x16_len[count] = sizeof(xx16[count]);
    x17_len[count] = sizeof(xx17[count]);
    x18_len[count] = sizeof(xx18[count]);
    x19_len[count] = sizeof(xx19[count]);

    count++;
  }

  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]), iconf_len);
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
    m_writeStmt->setDataBuffer(15, (dvoid*)xx13, OCCIINT, sizeof(xx13[0]), x13_len);
    m_writeStmt->setDataBuffer(16, (dvoid*)xx14, OCCIINT, sizeof(xx14[0]), x14_len);
    m_writeStmt->setDataBuffer(17, (dvoid*)xx15, OCCIINT, sizeof(xx15[0]), x15_len);
    m_writeStmt->setDataBuffer(18, (dvoid*)xx16, OCCIINT, sizeof(xx16[0]), x16_len);
    m_writeStmt->setDataBuffer(19, (dvoid*)xx17, OCCIINT, sizeof(xx17[0]), x17_len);
    m_writeStmt->setDataBuffer(20, (dvoid*)xx18, OCCIINT, sizeof(xx18[0]), x18_len);
    m_writeStmt->setDataBuffer(21, (dvoid*)xx19, OCCIINT, sizeof(xx19[0]), x19_len);

    m_writeStmt->executeArrayUpdate(nrows);

    delete[] ids;
    delete[] iconfid_vec;
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
    delete[] xx13;
    delete[] xx14;
    delete[] xx15;
    delete[] xx16;
    delete[] xx17;
    delete[] xx18;
    delete[] xx19;

    delete[] ids_len;
    delete[] iconf_len;
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
    delete[] x13_len;
    delete[] x14_len;
    delete[] x15_len;
    delete[] x16_len;
    delete[] x17_len;
    delete[] x18_len;
    delete[] x19_len;

  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigCokeDat::writeArrayDB():  " + e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightModeDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigOddWeightModeDat::FEConfigOddWeightModeDat() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_en_EB_flt = 0;
  m_en_EE_flt = 0;
  m_en_EB_pf = 0;
  m_en_EE_pf = 0;
  m_dis_EB_even_pf = 0;
  m_dis_EE_even_pf = 0;
  m_fe_EB_strout = 0;
  m_fe_EE_strout = 0;
  m_fe_EB_strib2 = 0;
  m_fe_EE_strib2 = 0;
  m_fe_EB_tcpout = 0;
  m_fe_EB_tcpib1 = 0;
  m_fe_EE_tcpout = 0;
  m_fe_EE_tcpib1 = 0;
  m_fe_par15 = 0;
  m_fe_par16 = 0;
  m_fe_par17 = 0;
  m_fe_par18 = 0;
}

FEConfigOddWeightModeDat::~FEConfigOddWeightModeDat() {}

void FEConfigOddWeightModeDat::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO " + getTable() +
        " (wei2_conf_id, "
        " enableEBOddFilter, enableEEOddFilter, enableEBOddPeakFinder,enableEEOddPeakFinder, disableEBEvenPeakFinder, "
        "DISABLEEEEVENPEAKFINDER, fenixEBStripOutput, fenixEEStripOutput, FenixEBStripInfoBit2, fenixEEStripInfobit2, "
        "EBfenixTcpOutput, EBfenixTCPInfobit1,EEFENIXTCPOUTPUT, EEFENIXTCPINFOBIT1 ,fenixpar15, fenixpar16, "
        "fenixpar17, fenixpar18  ) "
        "VALUES (:wei2_conf_id,  "
        " :w1, :w2, :w3, :w4, :w5, :w6, :w7, :w8, :w9, :w10, :w11, :w12, :w13, :w14, :w15 , :w16, :w17, :w18 )");
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightModeDat::prepareWrite():  " + e.getMessage()));
  }
}

void FEConfigOddWeightModeDat::writeDB(const EcalLogicID* ecid,
                                       const FEConfigOddWeightModeDat* item,
                                       FEConfigOddWeightInfo* iconf) noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigOddWeightModeDat::writeDB:  ICONF not in DB"));
  }

  try {
    m_writeStmt->setInt(1, iconfID);

    m_writeStmt->setInt(2, item->getFenixPar1());
    m_writeStmt->setInt(3, item->getFenixPar2());
    m_writeStmt->setInt(4, item->getFenixPar3());
    m_writeStmt->setInt(5, item->getFenixPar4());
    m_writeStmt->setInt(6, item->getFenixPar5());
    m_writeStmt->setInt(7, item->getFenixPar6());
    m_writeStmt->setInt(8, item->getFenixPar7());
    m_writeStmt->setInt(9, item->getFenixPar8());
    m_writeStmt->setInt(10, item->getFenixPar9());
    m_writeStmt->setInt(11, item->getFenixPar10());
    m_writeStmt->setInt(12, item->getFenixPar11());
    m_writeStmt->setInt(13, item->getFenixPar12());
    m_writeStmt->setInt(14, item->getFenixPar13());
    m_writeStmt->setInt(15, item->getFenixPar14());
    m_writeStmt->setInt(16, item->getFenixPar15());
    m_writeStmt->setInt(17, item->getFenixPar16());
    m_writeStmt->setInt(18, item->getFenixPar17());
    m_writeStmt->setInt(19, item->getFenixPar18());

    m_writeStmt->executeUpdate();
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightModeDat::writeDB():  " + e.getMessage()));
  }
}

void FEConfigOddWeightModeDat::fetchData(map<EcalLogicID, FEConfigOddWeightModeDat>* fillMap,
                                         FEConfigOddWeightInfo* iconf) noexcept(false) {
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) {
    throw(std::runtime_error("FEConfigOddWeightModeDat::fetchData:  ICONF not in DB"));
    return;
  }

  try {
    m_readStmt->setSQL(
        "SELECT  enableEBOddFilter, enableEEOddFilter, enableEBOddPeakFinder,enableEEOddPeakFinder, "
        "disableEBEvenPeakFinder, DISABLEEEEVENPEAKFINDER, fenixEBStripOutput, fenixEBStripOutput, "
        "FenixEBStripInfoBit2, fenixEEStripInfobit2, EBfenixTcpOutput, EBfenixTCPInfobit1,EEFENIXTCPOUTPUT, "
        "EEFENIXTCPINFOBIT1 ,fenixpar15, fenixpar16, fenixpar17, fenixpar18  "
        "FROM " +
        getTable() +
        " d "
        "WHERE wei2_conf_id = :wei2_conf_id ");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    std::pair<EcalLogicID, FEConfigOddWeightModeDat> p;
    FEConfigOddWeightModeDat dat;
    int ig = -1;
    while (rset->next()) {
      ig++;                              // we create a dummy logic_id
      p.first = EcalLogicID("Group_id",  // name
                            ig);         // logic_id

      dat.setFenixPar1(rset->getInt(1));
      dat.setFenixPar2(rset->getInt(2));
      dat.setFenixPar3(rset->getInt(3));
      dat.setFenixPar4(rset->getInt(4));
      dat.setFenixPar5(rset->getInt(5));
      dat.setFenixPar6(rset->getInt(6));
      dat.setFenixPar7(rset->getInt(7));
      dat.setFenixPar8(rset->getInt(8));
      dat.setFenixPar9(rset->getInt(9));
      dat.setFenixPar10(rset->getInt(10));
      dat.setFenixPar11(rset->getInt(11));
      dat.setFenixPar12(rset->getInt(12));
      dat.setFenixPar13(rset->getInt(13));
      dat.setFenixPar14(rset->getInt(14));
      dat.setFenixPar15(rset->getInt(15));
      dat.setFenixPar16(rset->getInt(16));
      dat.setFenixPar17(rset->getInt(17));
      dat.setFenixPar18(rset->getInt(18));

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("FEConfigOddWeightModeDat::fetchData:  " + e.getMessage()));
  }
}

void FEConfigOddWeightModeDat::writeArrayDB(const std::map<EcalLogicID, FEConfigOddWeightModeDat>* data,
                                            FEConfigOddWeightInfo* iconf) noexcept(false) {
  const EcalLogicID* channel;
  const FEConfigOddWeightModeDat* dataitem;

  typedef map<EcalLogicID, FEConfigOddWeightModeDat>::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    dataitem = &(p->second);
    writeDB(channel, dataitem, iconf);
  }
}

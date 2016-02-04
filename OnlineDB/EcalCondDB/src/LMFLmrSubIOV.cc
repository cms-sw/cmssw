#include "OnlineDB/EcalCondDB/interface/LMFLmrSubIOV.h"

void LMFLmrSubIOV::init() {
  m_className = "LMFLmrSubIOV";

  m_lmfIOV = 0;
  m_t[0] = Tm();
  m_t[1] = Tm();
  m_t[2] = Tm();
}

LMFLmrSubIOV::LMFLmrSubIOV() {
  init();
}

LMFLmrSubIOV::LMFLmrSubIOV(EcalDBConnection *c): LMFUnique(c) {
  init();
}

LMFLmrSubIOV::LMFLmrSubIOV(oracle::occi::Environment* env,
			   oracle::occi::Connection* conn): LMFUnique(env, conn)
{
  init();
}

LMFLmrSubIOV::~LMFLmrSubIOV() {
}

std::string LMFLmrSubIOV::fetchIdSql(Statement *stmt) {
  if (!m_lmfIOV) {
    if (m_debug) {
      std::cout << m_className << ": LMFIOV not set" << std::endl;
    }
    return "";
  }

  std::string sql = "SELECT LMR_SUB_IOV_ID FROM LMF_LMR_SUB_IOV "
    "WHERE "
    "IOV_ID  = :1 AND "
    "T1      = :2 AND "
    "T2      = :3 AND "
    "T3      = :4";
  stmt->setSQL(sql);
  stmt->setInt(1, m_lmfIOV);
  DateHandler dh(m_env, m_conn);
  for (int i = 0; i < 3; i++) {
    oracle::occi::Date t = dh.tmToDate(m_t[i]);
    stmt->setDate(i + 2, t);
  }
  return sql;

}

std::string LMFLmrSubIOV::setByIDSql(Statement *stmt, int id) {
  std::string sql = "SELECT IOV_ID, T1, T2, T3 FROM LMF_LMR_SUB_IOV "
    "WHERE LMR_SUB_IOV_ID = :1";
  stmt->setSQL(sql);
  stmt->setInt(1, id);
  return sql;
}

std::string LMFLmrSubIOV::writeDBSql(Statement *stmt) {
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  for (int i = 0; i < 3; i++) {
    if (m_t[i].isNull()) {
      m_t[i] = dh.getPlusInfTm();
    }
  }

  if (m_lmfIOV == 0) {
    throw(std::runtime_error(m_className + "::writeDB: LMFIOV not set"));
  }
  std::string sql = "INSERT INTO LMF_LMR_SUB_IOV (LMF_LMR_SUB_IOV_ID, "
    "IOV_ID, T1, T2, T3) "
    "VALUES (LMF_LMR_SUB_IOV_ID_SQ.NextVal, :1, :2, :3, :4)";
  stmt->setSQL(sql);
  stmt->setInt(1, m_lmfIOV);
  for (int i = 0; i < 3; i++) {
    stmt->setDate(i + 2, dh.tmToDate(m_t[i]));
  }
  return sql;
}

void LMFLmrSubIOV::getParameters(ResultSet *rset) {
  DateHandler dh(m_env, m_conn);
  m_lmfIOV = rset->getInt(1);
  for (int i = 0; i < 3; i++) {
    oracle::occi::Date t = rset->getDate(i + 2);
    m_t[i] = dh.dateToTm(t);
  }
}




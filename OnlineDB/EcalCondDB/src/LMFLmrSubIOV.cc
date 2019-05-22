#include "OnlineDB/EcalCondDB/interface/LMFLmrSubIOV.h"

#include <ctime>
#include <string>

void LMFLmrSubIOV::init() {
  m_className = "LMFLmrSubIOV";

  m_lmfIOV = 0;
  m_t[0] = Tm();
  m_t[1] = Tm();
  m_t[2] = Tm();
}

LMFLmrSubIOV::LMFLmrSubIOV() { init(); }

LMFLmrSubIOV::LMFLmrSubIOV(EcalDBConnection *c) : LMFUnique(c) { init(); }

LMFLmrSubIOV::LMFLmrSubIOV(oracle::occi::Environment *env, oracle::occi::Connection *conn) : LMFUnique(env, conn) {
  init();
}

LMFLmrSubIOV::~LMFLmrSubIOV() {}

LMFLmrSubIOV &LMFLmrSubIOV::setLMFIOV(const LMFIOV &iov) {
  if (m_debug) {
    std::cout << "[LMFLmrSubIOV] Setting IOV_ID as " << iov.getID() << std::endl << std::flush;
  }
  m_lmfIOV = iov.getID();
  return *this;
}

LMFLmrSubIOV &LMFLmrSubIOV::setTimes(Tm *t) {
  m_t[0] = t[0];
  m_t[1] = t[1];
  m_t[2] = t[2];
  return *this;
}

LMFLmrSubIOV &LMFLmrSubIOV::setTimes(const Tm &t1, const Tm &t2, const Tm &t3) {
  m_t[0] = t1;
  m_t[1] = t2;
  m_t[2] = t3;
  return *this;
}

std::string LMFLmrSubIOV::fetchIdSql(Statement *stmt) {
  if (!m_lmfIOV) {
    if (m_debug) {
      std::cout << m_className << ": LMFIOV not set" << std::endl;
    }
    return "";
  }

  std::string sql =
      "SELECT LMR_SUB_IOV_ID FROM "
      "CMS_ECAL_LASER_COND.LMF_LMR_SUB_IOV "
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
  std::string sql =
      "SELECT IOV_ID, T1, T2, T3 FROM "
      "CMS_ECAL_LASER_COND.LMF_LMR_SUB_IOV "
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
  std::string sp = sequencePostfix(m_t[0]);
  std::string sql =
      "INSERT INTO LMF_LMR_SUB_IOV (LMR_SUB_IOV_ID, "
      "IOV_ID, T1, T2, T3) "
      "VALUES (LMF_LMR_SUB_IOV_ID_" +
      sp + "_SQ.NextVal, :1, :2, :3, :4)";
  stmt->setSQL(sql);
  stmt->setInt(1, m_lmfIOV);
  for (int i = 0; i < 3; i++) {
    stmt->setDate(i + 2, dh.tmToDate(m_t[i]));
  }
  return sql;
}

void LMFLmrSubIOV::getParameters(ResultSet *rset) noexcept(false) {
  m_lmfIOV = rset->getInt(1);
  for (int i = 0; i < 3; i++) {
    oracle::occi::Date t = rset->getDate(i + 2);
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
    m_t[i].setToString(t.toText("YYYY-MM-DD HH24:MI:SS"));
#else
    int year = 0;
    unsigned int month = 0;
    unsigned int day = 0;
    unsigned int hour = 0;
    unsigned int min = 0;
    unsigned int seconds = 0;
    t.getDate(year, month, day, hour, min, seconds);
    const std::tm tt = {.tm_sec = static_cast<int>(seconds),
                        .tm_min = static_cast<int>(min),
                        .tm_hour = static_cast<int>(hour),
                        .tm_mday = static_cast<int>(day),
                        .tm_mon = static_cast<int>(month),
                        .tm_year = year - 1900};
    char tt_str[30] = {0};
    if (std::strftime(tt_str, sizeof(tt_str), "%F %T", &tt)) {
      m_t[i].setToString(std::string(tt_str));
    } else {
      throw std::runtime_error("LMFLmrSubIOV::writeDBSql: failed to generate the date string");
    }
#endif
  }
}

std::list<int> LMFLmrSubIOV::getIOVIDsLaterThan(const Tm &t) noexcept(false) {
  Tm tinf;
  tinf.setToString("9999-12-31 23:59:59");
  return getIOVIDsLaterThan(t, tinf, 0);
}

std::list<int> LMFLmrSubIOV::getIOVIDsLaterThan(const Tm &t, int howmany) noexcept(false) {
  Tm tinf;
  tinf.setToString("9999-12-31 23:59:59");
  return getIOVIDsLaterThan(t, tinf, howmany);
}

std::list<int> LMFLmrSubIOV::getIOVIDsLaterThan(const Tm &tmin, const Tm &tmax) noexcept(false) {
  return getIOVIDsLaterThan(tmin, tmax, 0);
}

std::list<int> LMFLmrSubIOV::getIOVIDsLaterThan(const Tm &tmin, const Tm &tmax, int howMany) noexcept(false) {
  Tm tinf;
  tinf.setToString("9999-12-31 23:59:59");
  std::string sql =
      "SELECT * FROM (SELECT LMR_SUB_IOV_ID "
      "FROM CMS_ECAL_LASER_COND.LMF_LMR_SUB_IOV WHERE T3 > :1 ";
  if (tmax != tinf) {
    sql += "AND T3 < :2 ORDER BY T3 ASC) ";
    if (howMany > 0) {
      sql += "WHERE ROWNUM <= :3";
    }
  } else {
    sql += "ORDER BY T3 ASC) ";
    if (howMany > 0) {
      sql += "WHERE ROWNUM <= :2";
    }
  }
  if (m_debug) {
    std::cout << "Executing query: " << std::endl << sql << std::endl;
  }
  std::list<int> ret;
  if (m_conn != nullptr) {
    try {
      DateHandler dh(m_env, m_conn);
      Statement *stmt = m_conn->createStatement();
      stmt->setPrefetchRowCount(10000);
      stmt->setSQL(sql);
      stmt->setDate(1, dh.tmToDate(tmin));
      if (tmax != tinf) {
        stmt->setDate(2, dh.tmToDate(tmax));
        if (howMany > 0) {
          stmt->setInt(3, howMany);
        }
      } else {
        if (howMany > 0) {
          stmt->setInt(2, howMany);
        }
      }
      ResultSet *rset = stmt->executeQuery();
      int row = 1;
      while (rset->next() != 0) {
        if (m_debug) {
          std::cout << "Getting row " << row++ << std::endl;
        }
        ret.push_back(rset->getInt(1));
      }
      stmt->setPrefetchRowCount(0);
      m_conn->terminateStatement(stmt);
    } catch (oracle::occi::SQLException &e) {
#if defined(_GLIBCXX_USE_CXX11_ABI) && (_GLIBCXX_USE_CXX11_ABI == 0)
      throw(std::runtime_error(m_className + "::getLmrSubIOVLaterThan: " + e.getMessage()));
#else
      throw(
          std::runtime_error(m_className + "::getLmrSubIOVLaterThan: error code " + std::to_string(e.getErrorCode())));
#endif
    }
  } else {
    throw(std::runtime_error(m_className + "::getLmrSubIOVLaterThan: " + "Connection not set"));
  }
  if (m_debug) {
    std::cout << "Sorting..." << std::flush;
  }
  ret.sort();
  if (m_debug) {
    std::cout << "Done!" << std::endl << std::flush;
  }
  return ret;
}

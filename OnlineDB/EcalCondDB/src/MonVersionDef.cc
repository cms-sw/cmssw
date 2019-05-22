#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonVersionDef.h"

using namespace std;
using namespace oracle::occi;

MonVersionDef::MonVersionDef() {
  m_env = nullptr;
  m_conn = nullptr;
  m_ID = 0;
  m_monVer = "";
  m_desc = "";
}

MonVersionDef::~MonVersionDef() {}

string MonVersionDef::getMonitoringVersion() const { return m_monVer; }

void MonVersionDef::setMonitoringVersion(string ver) {
  if (ver != m_monVer) {
    m_ID = 0;
    m_monVer = ver;
  }
}

string MonVersionDef::getDescription() const { return m_desc; }

int MonVersionDef::fetchID() noexcept(false) {
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT def_id FROM mon_version_def WHERE "
        "mon_ver = :1");
    stmt->setString(1, m_monVer);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MonVersionDef::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void MonVersionDef::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT mon_ver, description FROM mon_version_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_monVer = rset->getString(1);
      m_desc = rset->getString(2);
    } else {
      throw(std::runtime_error("MonVersionDef::setByID:  Given def_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MonVersionDef::setByID:  " + e.getMessage()));
  }
}

void MonVersionDef::fetchAllDefs(std::vector<MonVersionDef>* fillVec) noexcept(false) {
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM mon_version_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();

    MonVersionDef monVersionDef;
    monVersionDef.setConnection(m_env, m_conn);

    while (rset->next()) {
      monVersionDef.setByID(rset->getInt(1));
      fillVec->push_back(monVersionDef);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MonVersionDef::fetchAllDefs:  " + e.getMessage()));
  }
}

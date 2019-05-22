#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonRunOutcomeDef.h"

using namespace std;
using namespace oracle::occi;

MonRunOutcomeDef::MonRunOutcomeDef() {
  m_env = nullptr;
  m_conn = nullptr;
  m_ID = 0;
  m_shortDesc = "";
  m_longDesc = "";
}

MonRunOutcomeDef::~MonRunOutcomeDef() {}

string MonRunOutcomeDef::getShortDesc() const { return m_shortDesc; }

void MonRunOutcomeDef::setShortDesc(string desc) {
  if (desc != m_shortDesc) {
    m_ID = 0;
    m_shortDesc = desc;
  }
}

string MonRunOutcomeDef::getLongDesc() const { return m_longDesc; }

int MonRunOutcomeDef::fetchID() noexcept(false) {
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT def_id FROM mon_run_outcome_def WHERE "
        "short_desc = :1");
    stmt->setString(1, m_shortDesc);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MonRunOutcomeDef::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void MonRunOutcomeDef::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT short_desc, long_desc FROM mon_run_outcome_def WHERE def_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_shortDesc = rset->getString(1);
      m_longDesc = rset->getString(2);
      m_ID = id;
    } else {
      throw(std::runtime_error("MonRunOutcomeDef::setByID:  Given def_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("MonRunOutcomeDef::setByID:  " + e.getMessage()));
  }
}

void MonRunOutcomeDef::fetchAllDefs(std::vector<MonRunOutcomeDef>* fillVec) noexcept(false) {
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM mon_run_outcome_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();

    MonRunOutcomeDef def;
    def.setConnection(m_env, m_conn);

    while (rset->next()) {
      def.setByID(rset->getInt(1));
      fillVec->push_back(def);
    }
  } catch (SQLException& e) {
    throw(std::runtime_error("MonRunOutcomeDef::fetchAllDefs:  " + e.getMessage()));
  }
}

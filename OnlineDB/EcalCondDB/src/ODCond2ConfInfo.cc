#include <stdexcept>
#include <string>
#include <cstring>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

#include "OnlineDB/EcalCondDB/interface/ODCond2ConfInfo.h"

using namespace std;
using namespace oracle::occi;

ODCond2ConfInfo::ODCond2ConfInfo() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;
  m_config_tag = "";
  m_ID = 0;
  clear();
}

void ODCond2ConfInfo::clear() {
  m_type = "";
  m_loc = "";
  m_run = 0;
  m_desc = "";
  m_rec_time = Tm();
  m_db_time = Tm();
  m_typ_id = 0;
  m_loc_id = 0;
}

ODCond2ConfInfo::~ODCond2ConfInfo() {}

int ODCond2ConfInfo::fetchNextId() noexcept(false) {
  int result = 0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement();
    m_readStmt->setSQL("select COND2CONF_INFO_SQ.NextVal from DUAL ");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next()) {
      result = rset->getInt(1);
    }
    result++;
    m_conn->terminateStatement(m_readStmt);
    return result;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCond2ConfInfo::fetchNextId():  ") + e.getMessage()));
  }
}

void ODCond2ConfInfo::fetchParents() noexcept(false) {
  if (m_typ_id == 0) {
    if (!getType().empty()) {
      try {
        this->checkConnection();
        m_readStmt = m_conn->createStatement();
        m_readStmt->setSQL("select def_id from COND2CONF_TYPE_DEF where rec_type=" + getType());
        ResultSet* rset = m_readStmt->executeQuery();
        while (rset->next()) {
          m_typ_id = rset->getInt(1);
        }
        m_conn->terminateStatement(m_readStmt);

      } catch (SQLException& e) {
        throw(std::runtime_error(std::string("ODCond2ConfInfo::fetchParents():  ") + e.getMessage()));
      }
    }
  }
  if (m_loc_id == 0) {
    if (!getLocation().empty()) {
      try {
        this->checkConnection();
        m_readStmt = m_conn->createStatement();
        m_readStmt->setSQL("select def_id from location_def where location=" + getLocation());
        ResultSet* rset = m_readStmt->executeQuery();
        while (rset->next()) {
          m_loc_id = rset->getInt(1);
        }
        m_conn->terminateStatement(m_readStmt);
      } catch (SQLException& e) {
        throw(std::runtime_error(std::string("ODCond2ConfInfo::fetchParents():  ") + e.getMessage()));
      }
    }
  }
}

void ODCond2ConfInfo::prepareWrite() noexcept(false) {
  this->checkConnection();

  int next_id = 0;
  if (getId() == 0) {
    next_id = fetchNextId();
  }

  fetchParents();
  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() +
                        " ( rec_id, rec_type_id, rec_date, "
                        "location_id, run_number, short_desc ) "
                        " VALUES ( :1, :2, :3 , :4, :5, :6 ) ");

    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setInt(3, m_typ_id);
    m_writeStmt->setInt(4, m_loc_id);

    m_ID = next_id;

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCond2ConfInfo::prepareWrite():  ") + e.getMessage()));
  }
}

void ODCond2ConfInfo::setParameters(const std::map<string, string>& my_keys_map) {
  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for (std::map<std::string, std::string>::const_iterator ci = my_keys_map.begin(); ci != my_keys_map.end(); ci++) {
    //    if(ci->first==  "TAG") setConfigTag(ci->second);
  }
  std::cout << "method not yet implemented" << std::endl;
}

void ODCond2ConfInfo::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  DateHandler dh(m_env, m_conn);
  if (m_rec_time.isNull()) {
    int very_old_time = 0;
    m_rec_time = Tm(very_old_time);
  }

  try {
    m_writeStmt->setDate(3, dh.tmToDate(this->m_rec_time));
    m_writeStmt->setInt(5, this->getRunNumber());
    m_writeStmt->setString(6, this->getDescription());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCond2ConfInfo::writeDB():  ") + e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODCond2ConfInfo::writeDB:  Failed to write"));
  }
}

void ODCond2ConfInfo::fetchData(ODCond2ConfInfo* result) noexcept(false) {
  this->checkConnection();
  result->clear();

  DateHandler dh(m_env, m_conn);

  if (result->getId() == 0) {
    throw(std::runtime_error("ODCond2ConfInfo::fetchData(): no Id defined for this ODCond2ConfInfo "));
  }

  try {
    m_readStmt->setSQL(
        "SELECT rec_id, REC_TYPE, rec_date, location, "
        "run_number, short_desc, db_timestamp FROM " +
        getTable() +
        " , COND2CONF_TYPE_DEF , location_def "
        " where  rec_id = :1  AND COND2CONF_TYPE_DEF.def_id=" +
        getTable() + ".REC_TYPE_ID AND location_def.def_id=LOCATION_ID ");
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag and 3 is the version

    //    result->setId(rset->getInt(1));

    result->setType(rset->getString(2));
    Date startDate = rset->getDate(3);
    result->setLocation(rset->getString(4));
    result->setRunNumber(rset->getInt(5));
    result->setDescription(rset->getString(6));
    Date endDate = rset->getDate(7);

    m_rec_time = dh.dateToTm(startDate);
    m_db_time = dh.dateToTm(endDate);

  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCond2ConfInfo::fetchData():  ") + e.getMessage()));
  }
}

int ODCond2ConfInfo::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID != 0) {
    return m_ID;
  }

  this->checkConnection();

  fetchParents();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT rec_id FROM " + getTable() +
                 "WHERE  rec_type_id=:1 and (run_number=:2 or short_desc=:3 ) order by rec_id DESC ");

    stmt->setInt(1, m_typ_id);
    stmt->setInt(2, getRunNumber());
    stmt->setString(3, getDescription());

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error(std::string("ODCond2ConfInfo::fetchID:  ") + e.getMessage()));
  }

  return m_ID;
}

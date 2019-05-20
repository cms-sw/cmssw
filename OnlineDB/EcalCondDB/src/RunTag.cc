#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunTag.h"

using namespace std;
using namespace oracle::occi;

RunTag::RunTag() {
  m_env = nullptr;
  m_conn = nullptr;
  m_ID = 0;
  m_genTag = "default";
  m_locDef = LocationDef();
  m_runTypeDef = RunTypeDef();
}

RunTag::~RunTag() {}

string RunTag::getGeneralTag() const { return m_genTag; }

// User data methods

void RunTag::setGeneralTag(string genTag) {
  if (genTag != m_genTag) {
    m_ID = 0;
    m_genTag = genTag;
  }
}

LocationDef RunTag::getLocationDef() const { return m_locDef; }

void RunTag::setLocationDef(const LocationDef& locDef) {
  if (locDef != m_locDef) {
    m_ID = 0;
    m_locDef = locDef;
  }
}

RunTypeDef RunTag::getRunTypeDef() const { return m_runTypeDef; }

void RunTag::setRunTypeDef(const RunTypeDef& runTypeDef) {
  if (runTypeDef != m_runTypeDef) {
    m_ID = 0;
    m_runTypeDef = runTypeDef;
  }
}

int RunTag::fetchID() noexcept(false) {
  // Return tag from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  // fetch the parent IDs
  int locID, runTypeID;
  this->fetchParentIDs(&locID, &runTypeID);

  // fetch this ID
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT tag_id FROM run_tag WHERE "
        "gen_tag     = :1 AND "
        "location_id = :2 AND "
        "run_type_id = :3");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, locID);
    stmt->setInt(3, runTypeID);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("RunTag::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void RunTag::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT gen_tag, location_id, run_type_id FROM run_tag WHERE tag_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_genTag = rset->getString(1);
      int locID = rset->getInt(2);
      int runTypeID = rset->getInt(3);

      m_locDef.setConnection(m_env, m_conn);
      m_locDef.setByID(locID);

      m_runTypeDef.setConnection(m_env, m_conn);
      m_runTypeDef.setByID(runTypeID);

      m_ID = id;
    } else {
      throw(std::runtime_error("RunTag::setByID:  Given tag_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("RunTag::setByID:  " + e.getMessage()));
  }
}

int RunTag::writeDB() noexcept(false) {
  // see if this data is already in the DB
  if (this->fetchID()) {
    return m_ID;
  }

  // check the connectioin
  this->checkConnection();

  // fetch the parent IDs
  int locID, runTypeID;
  this->fetchParentIDs(&locID, &runTypeID);

  // write new tag to the DB
  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL(
        "INSERT INTO run_tag (tag_id, gen_tag, location_id, run_type_id) "
        "VALUES (run_tag_sq.NextVal, :1, :2, :3)");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, locID);
    stmt->setInt(3, runTypeID);

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("RunTag::writeDB:  " + e.getMessage()));
  }

  // now get the tag_id
  if (!this->fetchID()) {
    throw(std::runtime_error("RunTag::writeDB:  Failed to write"));
  }

  return m_ID;
}

void RunTag::fetchAllTags(std::vector<RunTag>* fillVec) noexcept(false) {
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM run_tag ORDER BY tag_id");
    ResultSet* rset = stmt->executeQuery();

    RunTag runtag;
    runtag.setConnection(m_env, m_conn);
    while (rset->next()) {
      runtag.setByID(rset->getInt(1));
      fillVec->push_back(runtag);
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("RunTag::fetchAllTags:  " + e.getMessage()));
  }
}

void RunTag::fetchParentIDs(int* locID, int* runTypeID) noexcept(false) {
  // get the location
  m_locDef.setConnection(m_env, m_conn);
  *locID = m_locDef.fetchID();

  // get the run type
  m_runTypeDef.setConnection(m_env, m_conn);
  *runTypeID = m_runTypeDef.fetchID();

  if (!*locID) {
    throw(std::runtime_error("RunTag::fetchparentids:  Given location does not exist in DB"));
  }

  if (!*runTypeID) {
    throw(std::runtime_error("RunTag::fetchParentIDs:  Given run type does not exist in DB"));
  }
}

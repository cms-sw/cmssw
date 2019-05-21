#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/CaliTag.h"

using namespace std;
using namespace oracle::occi;

CaliTag::CaliTag() {
  m_env = nullptr;
  m_conn = nullptr;
  m_ID = 0;
  m_genTag = "default";
  m_locDef = LocationDef();
  m_method = "default";
  m_version = "default";
  m_dataType = "default";
}

CaliTag::~CaliTag() {}

string CaliTag::getGeneralTag() const { return m_genTag; }

void CaliTag::setGeneralTag(string genTag) {
  if (genTag != m_genTag) {
    m_ID = 0;
    m_genTag = genTag;
  }
}

LocationDef CaliTag::getLocationDef() const { return m_locDef; }

void CaliTag::setLocationDef(const LocationDef& locDef) {
  if (locDef != m_locDef) {
    m_ID = 0;
    m_locDef = locDef;
  }
}

string CaliTag::getMethod() const { return m_method; }

void CaliTag::setMethod(string method) {
  if (method != m_method) {
    m_ID = 0;
    m_method = method;
  }
}

string CaliTag::getVersion() const { return m_version; }

void CaliTag::setVersion(string version) {
  if (version != m_version) {
    m_ID = 0;
    m_version = version;
  }
}

string CaliTag::getDataType() const { return m_dataType; }

void CaliTag::setDataType(string dataType) {
  if (dataType != m_dataType) {
    m_ID = 0;
    m_dataType = dataType;
  }
}

int CaliTag::fetchID() noexcept(false) {
  // Return tag from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  // fetch the parent IDs
  int locID;
  this->fetchParentIDs(&locID);

  // fetch this ID
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT tag_id FROM cali_tag WHERE "
        "gen_tag     = :1 AND "
        "location_id = :2 AND "
        "method      = :3 AND "
        "version     = :4 AND "
        "data_type    = :5");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, locID);
    stmt->setString(3, m_method);
    stmt->setString(4, m_version);
    stmt->setString(5, m_dataType);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliTag::fetchID:  " + e.getMessage()));
  }

  return m_ID;
}

void CaliTag::setByID(int id) noexcept(false) {
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL(
        "SELECT gen_tag, location_id, method, version, data_type "
        "FROM cali_tag WHERE tag_id = :1");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_genTag = rset->getString(1);
      int locID = rset->getInt(2);
      m_locDef.setConnection(m_env, m_conn);
      m_locDef.setByID(locID);
      m_method = rset->getString(3);
      m_version = rset->getString(4);
      m_dataType = rset->getString(5);

      m_ID = id;
    } else {
      throw(std::runtime_error("CaliTag::setByID:  Given tag_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliTag::setByID:  " + e.getMessage()));
  }
}

int CaliTag::writeDB() noexcept(false) {
  // see if this data is already in the DB
  if (this->fetchID()) {
    return m_ID;
  }

  // check the connectioin
  this->checkConnection();

  // fetch the parent IDs
  int locID;
  this->fetchParentIDs(&locID);

  // write new tag to the DB
  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL(
        "INSERT INTO cali_tag (tag_id, gen_tag, location_id, method, version, data_type) "
        "VALUES (cali_tag_sq.NextVal, :1, :2, :3, :4, :5)");
    stmt->setString(1, m_genTag);
    stmt->setInt(2, locID);
    stmt->setString(3, m_method);
    stmt->setString(4, m_version);
    stmt->setString(5, m_dataType);

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliTag::writeDB:  " + e.getMessage()));
  }

  // now get the tag_id
  if (!this->fetchID()) {
    throw(std::runtime_error("CaliTag::writeDB:  Failed to write"));
  }

  return m_ID;
}

void CaliTag::fetchAllTags(std::vector<CaliTag>* fillVec) noexcept(false) {
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT tag_id FROM cali_tag ORDER BY tag_id");
    ResultSet* rset = stmt->executeQuery();

    CaliTag dcutag;
    dcutag.setConnection(m_env, m_conn);
    while (rset->next()) {
      dcutag.setByID(rset->getInt(1));
      fillVec->push_back(dcutag);
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("CaliTag::fetchAllTags:  " + e.getMessage()));
  }
}

void CaliTag::fetchParentIDs(int* locID) noexcept(false) {
  // get the location
  m_locDef.setConnection(m_env, m_conn);
  *locID = m_locDef.fetchID();

  if (!*locID) {
    throw(std::runtime_error("CaliTag::writeDB:  Given location does not exist in DB"));
  }
}

#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODRunConfigSeqInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

ODRunConfigSeqInfo::ODRunConfigSeqInfo() {
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  m_ID = 0;
  //
  m_ecal_config_id = 0;
  m_seq_num = 0;
  m_cycles = 0;
  m_run_seq = RunSeqDef();
  m_description = "";
}

ODRunConfigSeqInfo::~ODRunConfigSeqInfo() {}

//
RunSeqDef ODRunConfigSeqInfo::getRunSeqDef() const { return m_run_seq; }
void ODRunConfigSeqInfo::setRunSeqDef(const RunSeqDef& run_seq) {
  if (run_seq != m_run_seq) {
    m_run_seq = run_seq;
  }
}
//

int ODRunConfigSeqInfo::fetchID() noexcept(false) {
  // Return from memory if available
  if (m_ID > 0) {
    return m_ID;
  }

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL(
        "SELECT sequence_id from ECAL_sequence_DAT "
        "WHERE ecal_config_id = :id1 "
        " and sequence_num = :id2  ");
    stmt->setInt(1, m_ecal_config_id);
    stmt->setInt(2, m_seq_num);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigSeqInfo::fetchID:  " + e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}

int ODRunConfigSeqInfo::fetchIDLast() noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(sequence_id) FROM ecal_sequence_dat ");
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigSeqInfo::fetchIDLast:  " + e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}

void ODRunConfigSeqInfo::setByID(int id) noexcept(false) {
  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  cout << "ODRunConfigSeqInfo::setByID called for id " << id << endl;

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL(
        "SELECT ecal_config_id, sequence_num, num_of_cycles, sequence_type_def_id, description FROM ECAL_sequence_DAT "
        "WHERE sequence_id = :1 ");
    stmt->setInt(1, id);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_ecal_config_id = rset->getInt(1);
      m_seq_num = rset->getInt(2);
      m_cycles = rset->getInt(3);
      int seq_def_id = rset->getInt(4);
      m_description = rset->getString(5);
      m_ID = id;
      m_run_seq.setConnection(m_env, m_conn);
      m_run_seq.setByID(seq_def_id);
    } else {
      throw(std::runtime_error("ODRunConfigSeqInfo::setByID:  Given config_id is not in the database"));
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigSeqInfo::setByID:  " + e.getMessage()));
  }
}

void ODRunConfigSeqInfo::prepareWrite() noexcept(false) {
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL(
        "INSERT INTO ECAL_SEQUENCE_DAT ( ecal_config_id, "
        "sequence_num, num_of_cycles, sequence_type_def_id, description ) "
        "VALUES (:1, :2, :3 , :4, :5 )");
  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigSeqInfo::prepareWrite():  " + e.getMessage()));
  }
}
void ODRunConfigSeqInfo::writeDB() noexcept(false) {
  this->checkConnection();
  this->checkPrepare();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {
    // get the run mode

    m_run_seq.setConnection(m_env, m_conn);
    int seq_def_id = m_run_seq.writeDB();

    m_writeStmt->setInt(1, this->getEcalConfigId());
    m_writeStmt->setInt(2, this->getSequenceNumber());
    m_writeStmt->setInt(3, this->getNumberOfCycles());
    m_writeStmt->setInt(4, seq_def_id);
    m_writeStmt->setString(5, this->getDescription());

    m_writeStmt->executeUpdate();

  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigSeqInfo::writeDB():  " + e.getMessage()));
  }
  if (!this->fetchID()) {
    throw(std::runtime_error("ODRunConfigSeqInfo::writeDB:  Failed to write"));
  }
  cout << "ODRunConfigSeqInfo::writeDB>> done inserting ODRunConfigSeqInfo with id=" << m_ID << endl;
}

void ODRunConfigSeqInfo::clear() {
  //  m_ecal_config_id =0;
  //  m_seq_num =0;
  m_ID = 0;
  m_cycles = 0;
  m_run_seq = RunSeqDef();
  m_description = "";
}

void ODRunConfigSeqInfo::fetchData(ODRunConfigSeqInfo* result) noexcept(false) {
  this->checkConnection();
  //  result->clear();
  if (result->getId() == 0) {
    //    throw(std::runtime_error("ODRunConfigSeqInfo::fetchData(): no Id defined for this record "));
    result->fetchID();
  }

  try {
    m_readStmt->setSQL(
        "SELECT ecal_config_id, sequence_num, num_of_cycles, "
        "sequence_type_def_id, description FROM ECAL_sequence_DAT WHERE sequence_id = :1 ");

    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setEcalConfigId(rset->getInt(1));
    result->setSequenceNumber(rset->getInt(2));
    result->setNumberOfCycles(rset->getInt(3));
    int seq_def_id = rset->getInt(4);

    m_run_seq.setConnection(m_env, m_conn);
    m_run_seq.setByID(seq_def_id);
    result->setDescription(rset->getString(5));

  } catch (SQLException& e) {
    throw(std::runtime_error("ODRunConfigSeqInfo::fetchData():  " + e.getMessage()));
  }
}

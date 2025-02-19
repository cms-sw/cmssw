#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/RunSeqDef.h"

using namespace std;
using namespace oracle::occi;

RunSeqDef::RunSeqDef()
{
  m_env = NULL;
  m_conn = NULL;
  m_ID = 0;
  m_runSeq = "";
  m_runType = RunTypeDef();
}



RunSeqDef::~RunSeqDef()
{
}



string RunSeqDef::getRunSeq() const {  return m_runSeq;}



void RunSeqDef::setRunSeq(string runseq){    m_runSeq = runseq;}

RunTypeDef RunSeqDef::getRunTypeDef() const
{
  return m_runType;
}

void RunSeqDef::setRunTypeDef(const RunTypeDef runTypeDef)
{
    m_runType = runTypeDef;
}


  
int RunSeqDef::fetchID()
  throw(std::runtime_error)
{
  // Return def from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();
  

  // get the run type
  m_runType.setConnection(m_env, m_conn);
  int run_type_id = m_runType.fetchID();


  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM ecal_sequence_type_def WHERE "
		 " run_type_def_id   = :1 and sequence_type_string = :2 "
		 );

    stmt->setInt(1, run_type_id);
    stmt->setString(2, m_runSeq);

    ResultSet* rset = stmt->executeQuery();
    
    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("RunSeqDef::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void RunSeqDef::setByID(int id) 
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("SELECT run_type_def_id, sequence_type_string FROM ecal_sequence_type_def WHERE def_id = :1");
    stmt->setInt(1, id);
    
    int idruntype=0;
    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      idruntype = rset->getInt(1);
      m_runSeq = rset->getString(2);
    } else {
      throw(std::runtime_error("RunSeqDef::setByID:  Given def_id is not in the database"));
    }

    m_conn->terminateStatement(stmt);

    m_runType.setConnection(m_env, m_conn);
    m_runType.setByID(idruntype);

  } catch (SQLException &e) {
   throw(std::runtime_error("RunSeqDef::setByID:  "+e.getMessage()));
  }
}



void RunSeqDef::fetchAllDefs( std::vector<RunSeqDef>* fillVec) 
  throw(std::runtime_error)
{
  this->checkConnection();
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT def_id FROM ecal_sequence_type_def ORDER BY def_id");
    ResultSet* rset = stmt->executeQuery();
    
    RunSeqDef runSeqDef;
    runSeqDef.setConnection(m_env, m_conn);

    while(rset->next()) {
      runSeqDef.setByID( rset->getInt(1) );
      fillVec->push_back( runSeqDef );
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("RunSeqDef::fetchAllDefs:  "+e.getMessage()));
  }
}

int RunSeqDef::writeDB()
  throw(std::runtime_error)
{
  // see if this data is already in the DB
  try {
    if (this->fetchID()) { 
      return m_ID; 
    }
  } catch (SQLException &e) {
    // it does not exist yet 
  }

  // check the connectioin
  this->checkConnection();

  // get the run type
  m_runType.setConnection(m_env, m_conn);
  int run_type_id = m_runType.fetchID();

  // write new seq def to the DB
  try {
    Statement* stmt = m_conn->createStatement();

    stmt->setSQL("insert into ecal_sequence_type_def(RUN_TYPE_DEF_ID, SEQUENCE_TYPE_STRING) values "
		 " ( :1, :2 )");
    stmt->setInt(1, run_type_id);
    stmt->setString(2, m_runSeq);
   

    stmt->executeUpdate();
    
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
   throw(std::runtime_error("RunSeqDef::writeDB:  "+e.getMessage()));
  }

  // now get the tag_id
  if (!this->fetchID()) {
    throw(std::runtime_error("RunSeqDef::writeDB:  Failed to write"));
  }

  return m_ID;
}


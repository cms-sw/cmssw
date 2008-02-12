#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODRunConfigSeqInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;


ODRunConfigSeqInfo::ODRunConfigSeqInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_ecal_config_id =0;
  m_seq_num =0;
  m_cycles =0;
  m_run_seq = RunSeqDef();
  m_description="";
}



ODRunConfigSeqInfo::~ODRunConfigSeqInfo(){}


void ODRunConfigSeqInfo::setID(int id){ m_ID = id;  }
int ODRunConfigSeqInfo::getID(){ return m_ID ;  }

void ODRunConfigSeqInfo::setDescription(std::string x) { m_description = x;}
std::string ODRunConfigSeqInfo::getDescription() const{  return m_description;}
//
void ODRunConfigSeqInfo::setNumberOfCycles(int n){ m_cycles = n;  }
int ODRunConfigSeqInfo::getNumberOfCycles()const {return m_cycles;  }
//
void ODRunConfigSeqInfo::setSequenceNumber(int n){ m_seq_num = n;  }
int ODRunConfigSeqInfo::getSequenceNumber()const {return m_seq_num;  }
//
RunSeqDef ODRunConfigSeqInfo::getRunSeqDef() const {  return m_run_seq;}
void ODRunConfigSeqInfo::setRunSeqDef(const RunSeqDef run_seq)
{
  if (run_seq != m_run_seq) {
    m_ID = 0;
    m_run_seq = run_seq;
  }
}
//




int ODRunConfigSeqInfo::fetchID()
  throw(runtime_error)
{
  // Return from memory if available
  if (m_ID>0) {
    return m_ID;
  }

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT sequence_id from ECAL_sequence_DAT "
		 "WHERE ecal_config_id = :id1 " 
		 " and sequence_num = :id2  " );
    stmt->setInt(1, m_ecal_config_id);
    stmt->setInt(2, m_seq_num);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigSeqInfo::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}



int ODRunConfigSeqInfo::fetchIDLast()
  throw(runtime_error)
{

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(sequence_id) FROM ecal_sequence_dat "	);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigSeqInfo::fetchIDLast:  "+e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}


void ODRunConfigSeqInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   cout<< "ODRunConfigSeqInfo::setByID called for id "<<id<<endl;

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT ecal_config_id, sequence_num, num_of_cycles, sequence_type_def, description FROM ECAL_sequence_DAT WHERE sequence_id = :1 ");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_ecal_config_id= rset->getInt(1);
       m_seq_num=rset->getInt(2);
       m_cycles=rset->getInt(3);
       int seq_def_id=rset->getInt(4);
       m_description= rset->getString(6);
       m_ID = id;
       m_run_seq.setConnection(m_env, m_conn);
       m_run_seq.setByID(seq_def_id);
     } else {
       throw(runtime_error("ODRunConfigSeqInfo::setByID:  Given config_id is not in the database"));
     }
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("ODRunConfigSeqInfo::setByID:  "+e.getMessage()));
   }
}



int ODRunConfigSeqInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {
    /*
    // in case we need to reserve an ID 
    Statement* stmtq = m_conn->createStatement();
    stmtq->setSQL("SELECT ecal_sequence_dat_sq.NextVal FROM dual "	);
    ResultSet* rsetq = stmtq->executeQuery();
    if (rsetq->next()) {
    m_ID = rsetq->getInt(1);
    } else {
    m_ID = 0;
    }
    m_conn->terminateStatement(stmtq);
    
    cout<< "ODRunConfigSeqInfo::writeDB>> going to use id "<<m_ID<<endl;
    */

    // get the run mode
    m_run_seq.setConnection(m_env, m_conn);
    int seq_def_id = m_run_seq.fetchID();

    // now insert 
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("INSERT INTO ECAL_SEQUENCE_DAT ( ecal_config_id, sequence_num, num_of_cycles, sequence_type_def, description ) "
     "VALUES (:1, :2, :3 , :4, :5 )");
   
    stmt->setInt(1, m_ecal_config_id );
    stmt->setInt(2, m_seq_num);
    stmt->setInt(3, m_cycles);
    stmt->setInt(4, seq_def_id);
    stmt->setString(5, m_description );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);

    fetchID();


  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigSeqInfo::writeDB:  "+e.getMessage()));
  }

  cout<< "ODRunConfigSeqInfo::writeDB>> done inserting ODRunConfigSeqInfo with id="<<m_ID<<endl;
  return m_ID;
}





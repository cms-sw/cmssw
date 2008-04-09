#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODRunConfigInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;


ODRunConfigInfo::ODRunConfigInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_tag ="";
  m_version=0;
  m_num_seq=0;
  m_runTypeDef = RunTypeDef();
  m_runModeDef = RunModeDef();
  m_defaults="";
  m_trigger_mode=""; 
  m_num_events=0;
}



ODRunConfigInfo::~ODRunConfigInfo(){}



//
RunTypeDef ODRunConfigInfo::getRunTypeDef() const {  return m_runTypeDef;}
void ODRunConfigInfo::setRunTypeDef(const RunTypeDef runTypeDef)
{
  if (runTypeDef != m_runTypeDef) {
    m_ID = 0;
    m_runTypeDef = runTypeDef;
  }
}
//
RunModeDef ODRunConfigInfo::getRunModeDef() const {  return m_runModeDef;}
void ODRunConfigInfo::setRunModeDef(const RunModeDef runModeDef)
{
  if (runModeDef != m_runModeDef) {
    m_ID = 0;
    m_runModeDef = runModeDef;
  }
}
//


int ODRunConfigInfo::fetchID()
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
    stmt->setSQL("SELECT config_id from ECAL_RUN_CONFIGURATION_DAT "
		 "WHERE tag = :tag " 
		 " and version = :version " );
    stmt->setString(1, m_tag);
    stmt->setInt(2, m_version);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigInfo::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}



int ODRunConfigInfo::fetchIDLast()
  throw(runtime_error)
{

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(config_id) FROM ecal_run_configuration_dat "	);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigInfo::fetchIDLast:  "+e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}

//
int ODRunConfigInfo::fetchIDFromTagAndVersion()
  throw(runtime_error)
{
  fetchID();
  return m_ID;
}



void ODRunConfigInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   cout<< "ODRunConfigInfo::setByID called for id "<<id<<endl;

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT tag, version, run_type_def_id, run_mode_def_id, num_of_sequences, description, defaults,"
		  " trg_mode,num_of_events, db_timestamp"
		  " FROM ECAL_RUN_CONFIGURATION_DAT WHERE config_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_tag= rset->getString(1);
       m_version= rset->getInt(2);
       int run_type_id=rset->getInt(3);
       int run_mode_id=rset->getInt(4);
       m_num_seq=rset->getInt(5);
       m_description= rset->getString(6);
       m_defaults= rset->getString(7);
       m_trigger_mode= rset->getString(8);
       m_num_events= rset->getInt(9);
       Date dbdate = rset->getDate(10);
       m_db_time = dh.dateToTm( dbdate );
       m_ID = id;
       m_runModeDef.setConnection(m_env, m_conn);
       m_runModeDef.setByID(run_mode_id);
       m_runTypeDef.setConnection(m_env, m_conn);
       m_runTypeDef.setByID(run_type_id);

     } else {
       throw(runtime_error("ODRunConfigInfo::setByID:  Given config_id is not in the database"));
     }
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("ODRunConfigInfo::setByID:  "+e.getMessage()));
   }
}

void ODRunConfigInfo::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_RUN_CONFIGURATION_DAT ( tag, version, run_type_def_id, "
		 " run_mode_def_id, num_of_sequences, defaults, trg_mode, num_of_events, description ) "
		 " VALUES (:1, :2, :3 , :4, :5, :6 ,:7, :8, :9 )");
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigInfo::prepareWrite():  "+e.getMessage()));
  }
}


void ODRunConfigInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {
    
    // get the run mode
    m_runModeDef.setConnection(m_env, m_conn);
    int run_mode_id = m_runModeDef.fetchID();

    // get the run type
    m_runTypeDef.setConnection(m_env, m_conn);
    int run_type_id = m_runTypeDef.fetchID();

    // now insert 

    m_writeStmt->setString(1, this->getTag());
    m_writeStmt->setInt(2, this->getVersion());
    m_writeStmt->setInt(3, run_type_id);
    m_writeStmt->setInt(4, run_mode_id);
    m_writeStmt->setInt(5, this->getNumberOfSequences());
    m_writeStmt->setString(6, this->getDefaults());
    m_writeStmt->setString(7, this->getTriggerMode());
    m_writeStmt->setInt(8, this->getNumberOfEvents());
    m_writeStmt->setString(9, this->getDescription());

    m_writeStmt->executeUpdate();

  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigInfo::writeDB:  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODRunConfigInfo::writeDB  Failed to write"));
  }

  cout<< "ODRunConfigInfo::writeDB>> done inserting ODRunConfigInfo with id="<<m_ID<<endl;
}





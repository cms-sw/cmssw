#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODRunConfigCycleInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;


ODRunConfigCycleInfo::ODRunConfigCycleInfo()
{
  m_conn = NULL;
  m_ID = 0;
  //
  m_sequence_id =0;
  m_cycle_num =0;
  m_tag = "";
  m_description="";
}



ODRunConfigCycleInfo::~ODRunConfigCycleInfo(){}


void ODRunConfigCycleInfo::setID(int id){ m_ID = id;  }
int ODRunConfigCycleInfo::getID(){ return m_ID ;  }

void ODRunConfigCycleInfo::setDescription(std::string x) { m_description = x;}
std::string ODRunConfigCycleInfo::getDescription() const{  return m_description;}
//
void ODRunConfigCycleInfo::setTag(std::string x) { m_tag = x;}
std::string ODRunConfigCycleInfo::getTag() const{  return m_tag;}
//
void ODRunConfigCycleInfo::setSequenceID(int x) { m_sequence_id = x;}
int ODRunConfigCycleInfo::getSequenceID() const{  return m_sequence_id;}
//
void ODRunConfigCycleInfo::setCycleNumber(int n){ m_cycle_num = n;  }
int ODRunConfigCycleInfo::getCycleNumber()const {return m_cycle_num;  }
//



int ODRunConfigCycleInfo::fetchID()
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
    stmt->setSQL("SELECT cycle_id from ECAL_cycle_DAT "
		 "WHERE sequence_id = :id1 " 
		 " and cycle_num = :id2  " );
    stmt->setInt(1, m_sequence_id);
    stmt->setInt(2, m_cycle_num);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigCycleInfo::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}



int ODRunConfigCycleInfo::fetchIDLast()
  throw(runtime_error)
{

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(cycle_id) FROM ecal_cycle_dat "	);
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigCycleInfo::fetchIDLast:  "+e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}


void ODRunConfigCycleInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   cout<< "ODRunConfigCycleInfo::setByID called for id "<<id<<endl;

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT sequence_id , cycle_num , tag , description FROM ECAL_cycle_DAT WHERE cycle_id = :1 ");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_sequence_id=rset->getInt(1);
       m_cycle_num=rset->getInt(2);
       m_tag = rset->getString(3);
       m_description= rset->getString(4);
       m_ID = id;
     } else {
       throw(runtime_error("ODRunConfigCycleInfo::setByID:  Given cycle_id is not in the database"));
     }
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(runtime_error("ODRunConfigCycleInfo::setByID:  "+e.getMessage()));
   }
}



int ODRunConfigCycleInfo::writeDB()
  throw(runtime_error)
{
  this->checkConnection();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {

    // now insert 
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("INSERT INTO ECAL_CYCLE_DAT ( sequence_id , cycle_num, tag, description ) "
     "VALUES (:1, :2, :3 , :4 )");
   
    stmt->setInt(1, m_sequence_id);
    stmt->setInt(2, m_cycle_num);
    stmt->setString(3, m_tag);
    stmt->setString(4, m_description );

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);

    fetchID();


  } catch (SQLException &e) {
    throw(runtime_error("ODRunConfigCycleInfo::writeDB:  "+e.getMessage()));
  }

  cout<< "ODRunConfigCycleInfo::writeDB>> done inserting ODRunConfigCycleInfo with id="<<m_ID<<endl;
  return m_ID;
}





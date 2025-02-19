#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;

DCUIOV::DCUIOV()
{
  m_conn = NULL;
  m_ID = 0;
  m_since = Tm();
  m_till = Tm();
}



DCUIOV::~DCUIOV()
{
}



void DCUIOV::setSince(Tm since)
{
  if (since != m_since) {
    m_ID = 0;
    m_since = since;
  }
}



Tm DCUIOV::getSince() const
{
  return m_since;
}



void DCUIOV::setTill(Tm till)
{
  if (till != m_till) {
    m_ID = 0;
    m_till = till;
  }
}



Tm DCUIOV::getTill() const
{
  return m_till;
}



void DCUIOV::setDCUTag(DCUTag tag)
{
  if (tag != m_dcuTag) {
    m_ID = 0;
    m_dcuTag = tag;
  }
}



DCUTag DCUIOV::getDCUTag() const
{
  return m_dcuTag;
}



int DCUIOV::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID) {
    return m_ID;
  }

  this->checkConnection();

  m_dcuTag.setConnection(m_env, m_conn);
  int tagID = m_dcuTag.fetchID();
  if (!tagID) { 
    return 0;
  }

  DateHandler dh(m_env, m_conn);

  if (m_till.isNull()) {
    m_till = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT iov_id FROM dcu_iov "
		 "WHERE tag_id = :tag_id AND "
		 "since = :since ");
    stmt->setInt(1, tagID);
    stmt->setDate(2, dh.tmToDate(m_since));
    //    stmt->setDate(3, dh.tmToDate(m_till));
  
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUIOV::fetchID:  "+e.getMessage()));
  }

  return m_ID;
}



void DCUIOV::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT tag_id, since, till FROM dcu_iov WHERE iov_id = :1");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       int tagID = rset->getInt(1);
       Date since = rset->getDate(2);
       Date till = rset->getDate(3);
	 
       m_since = dh.dateToTm( since );
       m_till = dh.dateToTm( till );

       m_dcuTag.setConnection(m_env, m_conn);
       m_dcuTag.setByID(tagID);
       m_ID = id;
     } else {
       throw(std::runtime_error("DCUTag::setByID:  Given tag_id is not in the database"));
     }
     
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("DCUTag::setByID:  "+e.getMessage()));
   }
}



int DCUIOV::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();

  // Check if this IOV has already been written
  if (this->fetchID()) {
    return m_ID;
  }

  m_dcuTag.setConnection(m_env, m_conn);
  int tagID = m_dcuTag.writeDB();
  
  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  if (m_since.isNull()) {
    throw(std::runtime_error("DCUIOV::writeDB:  Must setSince before writing"));
  }
  
  if (m_till.isNull()) {
    m_till = dh.getPlusInfTm();
  }

  try {
    Statement* stmt = m_conn->createStatement();
    
    stmt->setSQL("INSERT INTO dcu_iov (iov_id, tag_id, since, till) "
		 "VALUES (dcu_iov_sq.NextVal, :1, :2, :3)");
    stmt->setInt(1, tagID);
    stmt->setDate(2, dh.tmToDate(m_since));
    stmt->setDate(3, dh.tmToDate(m_till));

    stmt->executeUpdate();

    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUIOV::writeDB:  "+e.getMessage()));
  }

  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("DCUIOV::writeDB:  Failed to write"));
  }
  
  return m_ID;
}



void DCUIOV::setByTm(DCUTag* tag, Tm eventTm)
  throw(std::runtime_error)
{
  this->checkConnection();
  
  tag->setConnection(m_env, m_conn);
  int tagID = tag->fetchID();
  
  if (!tagID) {
    throw(std::runtime_error("DCUIOV::setByTm:  Given DCUTag does not exist in the DB"));
  }

  DateHandler dh(m_env, m_conn);

  Date eventDate = dh.tmToDate(eventTm);

  try {
    Statement* stmt = m_conn->createStatement();


    stmt->setSQL("SELECT iov_id, since, till FROM dcu_iov "
		 "WHERE tag_id = :1 AND since <= :2 AND till > :3");
    stmt->setInt(1, tagID);
    stmt->setDate(2, eventDate);
    stmt->setDate(3, eventDate);

    ResultSet* rset = stmt->executeQuery();
    if (rset->next()) {
      m_dcuTag = *tag;
      
      m_ID = rset->getInt(1);
      Date sinceDate = rset->getDate(2);
      Date tillDate = rset->getDate(3);
	 
      m_since = dh.dateToTm( sinceDate );
      m_till = dh.dateToTm( tillDate );
    } else {
      throw(std::runtime_error("DCUIOV::setByTm:  Given subrun is not in the database"));
    }
     
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUIOV::setByTm:  "+e.getMessage()));
  }
}

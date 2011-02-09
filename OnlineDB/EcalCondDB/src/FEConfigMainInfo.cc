#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

using namespace std;
using namespace oracle::occi;


FEConfigMainInfo::FEConfigMainInfo()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_ID=0;
  m_version=0;
  clear();

}



FEConfigMainInfo::~FEConfigMainInfo(){}


void FEConfigMainInfo::clear() {

  m_description="";
  m_ped_id=0;
  m_lin_id=0;
  m_lut_id=0;
  m_sli_id=0;
  m_fgr_id=0;
  m_wei_id=0;
  m_bxt_id=0;
  m_btt_id=0;
  m_db_time=Tm();


}
int FEConfigMainInfo::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select fe_config_main_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigMainInfo::fetchNextId():  "+e.getMessage()));
  }

}

int FEConfigMainInfo::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID>0) {
    return m_ID;
  }

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

 std::cout << " tag/version " << getConfigTag() <<"/"<<getVersion() << std::endl;

  try {
    Statement* stmt = m_conn->createStatement();
    if(m_version !=0){
      stmt->setSQL("SELECT conf_id from FE_CONFIG_MAIN "
		   "WHERE tag = :tag " 
		   " and version = :version " );
      stmt->setString(1, m_config_tag);
      stmt->setInt(2, m_version);
      std::cout<<" using query with version " <<endl;
    } else {
      // always select the last inserted one with a given tag
      stmt->setSQL("SELECT conf_id from FE_CONFIG_MAIN "
		   "WHERE tag = :1 and version= (select max(version) from FE_CONFIG_MAIN where tag=:2) " );
      stmt->setString(1, m_config_tag);
      stmt->setString(2, m_config_tag);
      std::cout<<" using query WITHOUT version " <<endl;
    }

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    std::cout<<m_ID<<endl;
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigMainInfo::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}




void FEConfigMainInfo::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();


  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO fe_config_main (conf_id, ped_conf_id, lin_conf_id, lut_conf_id, fgr_conf_id, sli_conf_id, wei_conf_id, bxt_conf_id, btt_conf_id, tag, version, description) "
			" VALUES (:1, :2, :3 , :4, :5, :6 ,:7, :8, :9, :10, :11, :12 )");

    m_writeStmt->setInt(1, next_id);
    m_ID=next_id;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigMainInfo::prepareWrite():  "+e.getMessage()));
  }

}


void FEConfigMainInfo::writeDB()
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  // Validate the data, use infinity-till convention
  DateHandler dh(m_env, m_conn);

  try {


    m_writeStmt->setInt(2, this->getPedId());
    m_writeStmt->setInt(3, this->getLinId());
    m_writeStmt->setInt(4, this->getLUTId());
    m_writeStmt->setInt(5, this->getFgrId());
    m_writeStmt->setInt(6, this->getSliId());
    m_writeStmt->setInt(7, this->getWeiId());
    m_writeStmt->setInt(8, this->getBxtId());
    m_writeStmt->setInt(9, this->getBttId());
    m_writeStmt->setString(10, this->getConfigTag());
    m_writeStmt->setInt(11, this->getVersion());
    m_writeStmt->setString(12, this->getDescription());
    m_writeStmt->executeUpdate();


  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigMainInfo::writeDB:  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("FEConfigMainInfo::writeDB:  Failed to write"));
  }
  setByID(m_ID);

  cout<< "FEConfigMainInfo::writeDB>> done inserting FEConfigMainInfo with id="<<m_ID<<endl;

}




int FEConfigMainInfo::fetchIDLast()
  throw(std::runtime_error)
{

  this->checkConnection();

  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT max(conf_id) FROM fe_config_main ");
    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODRunConfigInfo::fetchIDLast:  "+e.getMessage()));
  }

  setByID(m_ID);
  return m_ID;
}


void FEConfigMainInfo::setByID(int id) 
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   cout<< "FEConfigMainInfo::setByID called for id "<<id<<endl;

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT conf_id, ped_conf_id, lin_conf_id, lut_conf_id, fgr_conf_id, sli_conf_id, wei_conf_id,\
 bxt_conf_id, btt_conf_id, tag, version, description, db_timestamp FROM FE_CONFIG_MAIN WHERE conf_id = :1 ");
     stmt->setInt(1, id);
     
     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       setId(          rset->getInt(1) );
       setPedId(       rset->getInt(2) );
       setLinId(       rset->getInt(3) );
       setLUTId(       rset->getInt(4) );
       setFgrId(       rset->getInt(5) );
       setSliId(       rset->getInt(6) );
       setWeiId(       rset->getInt(7) );
       setBxtId(       rset->getInt(8) );
       setBttId(       rset->getInt(9) );
       setConfigTag(   rset->getString(10) );
       setVersion(     rset->getInt(11) );
       setDescription(      rset->getString(12) );
       Date dbdate = rset->getDate(13);
       setDBTime( dh.dateToTm( dbdate ));
       m_ID = id;
     } else {
       throw(std::runtime_error("FEConfigMainInfo::setByID:  Given cycle_id is not in the database"));
     }
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("FEConfigMainInfo::setByID:  "+e.getMessage()));
   }
}


void FEConfigMainInfo::fetchData(FEConfigMainInfo * result)
  throw(std::runtime_error)
{ std::cout << " ### 1 getId from FEConfigMainInfo = " << result->getId() << std::endl;
 std::cout << " tag/version " << result->getConfigTag() <<"/"<<result->getVersion() << std::endl;
  
  this->checkConnection();
   DateHandler dh(m_env, m_conn);
   //   result->clear();

  int idid=0;

  if(result->getId()==0){  
    //throw(std::runtime_error("FEConfigMainInfo::fetchData(): no Id defined for this FEConfigMainInfo "));
    idid=result->fetchID();
    result->setId(idid);
  }

  try {
    m_readStmt->setSQL("SELECT conf_id, ped_conf_id, lin_conf_id, lut_conf_id, fgr_conf_id, sli_conf_id, wei_conf_id, bxt_conf_id, btt_conf_id, tag, version, description, db_timestamp FROM FE_CONFIG_MAIN WHERE conf_id = :1 ");

    std::cout << " ### 2 getId from FEConfigMainInfo = " << result->getId() << std::endl;
     
    // good m_readStmt->setInt(1, result->getId());
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(          rset->getInt(1) );
    std::cout << " Id = " << rset->getInt(1) << std::endl;
    result->setPedId(       rset->getInt(2) );
    std::cout << " PedId = " << rset->getInt(2) << std::endl;
    result->setLinId(       rset->getInt(3) );
    result->setLUTId(       rset->getInt(4) );
    result->setFgrId(       rset->getInt(5) );
    result->setSliId(       rset->getInt(6) );
    result->setWeiId(       rset->getInt(7) );
    result->setBxtId(       rset->getInt(8) );
    result->setBttId(       rset->getInt(9) );
    result->setConfigTag(         rset->getString(10) );
    result->setVersion(     rset->getInt(11) );
    result->setDescription(      rset->getString(12) );
    Date dbdate = rset->getDate(13);
    result->setDBTime( dh.dateToTm( dbdate ));
 
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigMainInfo::fetchData():  "+e.getMessage()));
  }
}

 void FEConfigMainInfo::insertConfig()
  throw(std::runtime_error)
{
  try {

    prepareWrite();
    writeDB();
    m_conn->commit();
    terminateWriteStatement();
  } catch (std::runtime_error &e) {
    m_conn->rollback();
    throw(e);
  } catch (...) {
    m_conn->rollback();
    throw(std::runtime_error("FEConfigMainInfo::insertConfig:  Unknown exception caught"));
  }
}


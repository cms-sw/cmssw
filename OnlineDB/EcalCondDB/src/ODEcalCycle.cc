#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"

#include "OnlineDB/EcalCondDB/interface/ODEcalCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODCCSCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODDCCCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODJBH4Cycle.h"
#include "OnlineDB/EcalCondDB/interface/ODLaserCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODLTCCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODLTSCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODDCUCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODRunConfigCycleInfo.h"
#include "OnlineDB/EcalCondDB/interface/ODScanCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODTCCCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODTCCEECycle.h"
#include "OnlineDB/EcalCondDB/interface/ODTTCciCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODTTCFCycle.h"
#include "OnlineDB/EcalCondDB/interface/ODSRPCycle.h"

using namespace std;
using namespace oracle::occi;

ODEcalCycle::ODEcalCycle()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

   m_ID=0;
   clear();
}



ODEcalCycle::~ODEcalCycle()
{
}



void ODEcalCycle::prepareWrite()
  throw(std::runtime_error)
{

  std::cout<< "ODEcalCycle::prepareWrite(): this is a view writing specific tables  "<< endl;

}



void ODEcalCycle::writeDB()
  throw(std::runtime_error)
{

  ODRunConfigCycleInfo cyc;
  cyc.setTag(this->getCycleTag());
  cyc.setDescription(this->getCycleDescription());
  cyc.setCycleNumber(this->getCycleNum());
  cyc.setSequenceID(this->getSequenceId());
  // here we insert the cycle and get back the id
  cyc.setConnection(m_env, m_conn);
  cyc.insertConfig();
  int cyc_id=cyc.getId();
  cout << "Cycle inserted with ID "<< cyc_id<< endl;
  m_ID=cyc_id;

  if(getCCSId()!=0){ 
    ODCCSCycle ccs_cycle;
    ccs_cycle.setId(cyc_id);
    ccs_cycle.setCCSConfigurationID(getCCSId());
    ccs_cycle.setConnection(m_env, m_conn);
    ccs_cycle.insertConfig();
    cout << "Inserting CCS-cycle in DB..." << flush;
  } 
  if(getDCCId()!=0){
    ODDCCCycle dcc_cycle;
    dcc_cycle.setId(cyc_id);
    dcc_cycle.setDCCConfigurationID(getDCCId());
    dcc_cycle.setConnection(m_env, m_conn);
    dcc_cycle.insertConfig();
    cout << "Inserting DCC-cycle in DB..." << flush;
  }
  if(getLaserId()!=0){
    ODLaserCycle laser_cycle;
    laser_cycle.setId(cyc_id);
    laser_cycle.setLaserConfigurationID(getLaserId());
    laser_cycle.setConnection(m_env, m_conn);
    laser_cycle.insertConfig();
    cout << "Inserting LASER-cycle in DB..." << flush;
  }
  if(getLTCId()!=0){
    ODLTCCycle ltc_cycle;
    ltc_cycle.setId(cyc_id);
    ltc_cycle.setLTCConfigurationID(getLTCId());
    ltc_cycle.setConnection(m_env, m_conn);
    ltc_cycle.insertConfig();
    cout << "Inserting LTC-cycle in DB..." << flush;
  }
  if(getLTSId()!=0){
    ODLTSCycle lts_cycle;
    lts_cycle.setId(cyc_id);
    lts_cycle.setLTSConfigurationID(getLTSId());
    lts_cycle.setConnection(m_env, m_conn);
    lts_cycle.insertConfig();
    cout << "Inserting LTS-cycle in DB..." << flush;
  }
  if(getDCUId()!=0){
    ODDCUCycle dcu_cycle;
    dcu_cycle.setId(cyc_id);
    dcu_cycle.setDCUConfigurationID(getDCUId());
    dcu_cycle.setConnection(m_env, m_conn);
    dcu_cycle.insertConfig();
    cout << "Inserting DCU-cycle in DB..." << flush;
  }
  if(getTCCId()!=0){
    ODTCCCycle tcc_cycle;
    tcc_cycle.setId(cyc_id);
    tcc_cycle.setTCCConfigurationID(getTCCId());
    tcc_cycle.setConnection(m_env, m_conn);
    tcc_cycle.insertConfig();
    cout << "Inserting TCC-cycle in DB..." << flush;
  }
  if(getTCCEEId()!=0){
    ODTCCEECycle tcc_ee_cycle;
    tcc_ee_cycle.setId(cyc_id);
    tcc_ee_cycle.setTCCConfigurationID(getTCCEEId());
    tcc_ee_cycle.setConnection(m_env, m_conn);
    tcc_ee_cycle.insertConfig();
    cout << "Inserting TCCEE-cycle in DB..." << flush;
  }
  if(getTTCCIId()!=0){
    ODTTCciCycle ttcci_cycle;
    ttcci_cycle.setId(cyc_id);
    ttcci_cycle.setTTCciConfigurationID(getTTCCIId());
    ttcci_cycle.setConnection(m_env, m_conn);
    ttcci_cycle.insertConfig();
    cout << "Inserting TTCCI-cycle in DB..." << flush;
  }
  if(getTTCFId()!=0){
    ODTTCFCycle ttcf_cycle;
    ttcf_cycle.setId(cyc_id);
    ttcf_cycle.setTTCFConfigurationID(getTTCFId());
    ttcf_cycle.setConnection(m_env, m_conn);
    ttcf_cycle.insertConfig();
    cout << "Inserting TTCF-cycle in DB..." << flush;
  }
  if(getSRPId()!=0){
    ODSRPCycle srp_cycle;
    srp_cycle.setId(cyc_id);
    srp_cycle.setSRPConfigurationID(getSRPId());
    srp_cycle.setConnection(m_env, m_conn);
    srp_cycle.insertConfig();
    cout << "Inserting SRP-cycle in DB..." << flush;
  }
  if(getJBH4Id()!=0){
    ODJBH4Cycle jbh4_cycle;
    jbh4_cycle.setId(cyc_id);
    jbh4_cycle.setJBH4ConfigurationID(getJBH4Id());
    jbh4_cycle.setConnection(m_env, m_conn);
    jbh4_cycle.insertConfig();
    cout << "Inserting JBH4-cycle in DB..." << flush;
  }
  if(getScanId()!=0){
    ODScanCycle scan_cycle;
    scan_cycle.setId(cyc_id);
    scan_cycle.setScanConfigurationID(getScanId());
    scan_cycle.setConnection(m_env, m_conn);
    scan_cycle.insertConfig();
    cout << "Inserting SCAN-cycle in DB..." << flush;
  }


}


void ODEcalCycle::clear(){
   m_tag="";
   m_version=0;
   m_seq_num=0;
   m_cycle_num=0;
   m_cycle_tag="";
   m_cycle_description="";
   m_ccs=0;
   m_dcc=0;
   m_laser=0;
   m_ltc=0;
   m_lts=0;
   m_dcu=0;
   m_tcc=0;
   m_tcc_ee=0;
   m_ttcci=0;
   m_jbh4=0;
   m_scan=0;
   //   m_seq_id=0;
   m_ttcf=0;
   m_srp=0;
}

int ODEcalCycle::fetchID()
  throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID>0) {
    return m_ID;
  }

  this->checkConnection();


  DateHandler dh(m_env, m_conn);

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT cycle_id from ECAL_CYCLE_DAT "
               "WHERE sequence_id = :1 ");
    stmt->setInt(1, m_seq_id);

    ResultSet* rset = stmt->executeQuery();

    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODEcalCycle::fetchID:  "+e.getMessage()));
  }
  setByID(m_ID);
  return m_ID;
}

void ODEcalCycle::setByID(int id)
  throw(std::runtime_error)
{
   this->checkConnection();

   DateHandler dh(m_env, m_conn);

   try {
     Statement* stmt = m_conn->createStatement();

     stmt->setSQL("SELECT tag, version, sequence_num, cycle_num, cycle_tag, description, "
                     "ccs_configuration_id, dcc_configuration_id, laser_configuration_id, "
                     "ltc_configuration_id, lts_configuration_id, tcc_configuration_id, "
                     "\"TTCci_CONFIGURATION_ID\", jbh4_configuration_id, scan_id, TTCF_configuration_id, "
                     "srp_configuration_id, dcu_configuration_id, tcc_ee_configuration_id "
                     "FROM ECAL_CYCLE where cycle_id = :1 " );
     stmt->setInt(1, id);

     ResultSet* rset = stmt->executeQuery();
     if (rset->next()) {
       m_tag = rset->getString(1);
       m_version = rset->getInt(2);
       m_seq_num = rset->getInt(3);
       m_cycle_num = rset->getInt(4);
       m_cycle_tag = rset->getString(5);
       m_cycle_description = rset->getString(6);
       m_ccs = rset->getInt(7);
       m_dcc = rset->getInt(8);
       m_laser = rset->getInt(9);
       m_ltc = rset->getInt(10);
       m_lts = rset->getInt(11);
       m_tcc = rset->getInt(12);
       m_ttcci = rset->getInt(13);
       m_jbh4 = rset->getInt(14);
       m_scan = rset->getInt(15);
       m_ttcf = rset->getInt(16);
       m_srp = rset->getInt(17);
       m_dcu = rset->getInt(18);
       m_tcc_ee = rset->getInt(19);

     } else {
       throw(std::runtime_error("ODEcalCycle::setByID:  Given config_id is not in the database"));
     }
     m_conn->terminateStatement(stmt);
   } catch (SQLException &e) {
     throw(std::runtime_error("ODEcalCycle::setByID:  "+e.getMessage()));
   }
}


void ODEcalCycle::printout(){

  std::cout << "**** ODEcalCycle ****"<< std::endl;
  std::cout << "**** id: "<< m_ID << std::endl;
  std::cout << "**** tag: "<< m_tag << std::endl;
  std::cout << "**** version: "<< m_version << std::endl;
  std::cout << "**** seq_num: "<< m_seq_num << std::endl;
  std::cout << "**** cycle num: "<< m_cycle_num << std::endl;
  std::cout << "**** cycle tag: "<< m_cycle_tag << std::endl;
  std::cout << "**** cycle description: "<< m_cycle_description<< std::endl;
  std::cout << "**** ccs_id: "<< m_ccs <<std::endl;
  std::cout << "**** dcc_id: "<< m_dcc <<std::endl;
  std::cout << "**** dcu_id: "<< m_dcu <<std::endl;
  std::cout << "**** jbh4_id: "<< m_jbh4<<std::endl;
  std::cout << "**** ltc_id: "<< m_ltc <<std::endl;
  std::cout << "**** lts_id: "<< m_lts <<std::endl;
  std::cout << "**** laser: "<< m_laser << std::endl;
  std::cout << "**** ttcf_id: "<< m_ttcf << std::endl;
  std::cout << "**** tcc_id: "<< m_tcc << std::endl;
  std::cout << "**** tccee_id: "<< m_tcc_ee << std::endl;
  std::cout << "**** ttcci_id: "<< m_ttcci << std::endl;
  std::cout << "**** ttcf_id: "<< m_ttcf << std::endl;
  std::cout << "**** srp_id: "<< m_srp << std::endl;
  std::cout << "**** scan_id: "<< m_scan << std::endl;
  std::cout << "**** ODEcalCycle ****"<< std::endl;
 
 
}


void ODEcalCycle::fetchData(ODEcalCycle * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  //  result->clear();
  if(result->getId()==0){
    //    throw(std::runtime_error("ODEcalCycle::fetchData(): no Id defined for this ODEcalCycle "));
    result->fetchID();
  }

  try {

    m_readStmt->setSQL("SELECT tag, version, sequence_num, cycle_num, cycle_tag, description, "
                     "ccs_configuration_id, dcc_configuration_id, laser_configuration_id, "
                     "ltc_configuration_id, lts_configuration_id, tcc_configuration_id, "
                     "\"TTCci_CONFIGURATION_ID\", jbh4_configuration_id, scan_id, TTCF_configuration_id, "
                     "srp_configuration_id, dcu_configuration_id, tcc_ee_configuration_id "
                     "FROM ECAL_CYCLE where cycle_id = :1 " );
    m_readStmt->setInt(1, result->getId());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setTag(          rset->getString(1) );
    result->setVersion(      rset->getInt(2) );
    result->setSeqNum(       rset->getInt(3) );
    result->setCycleNum(     rset->getInt(4) );
    result->setCycleTag(     rset->getString(5) );
    result->setCycleDescription( rset->getString(6) );
    result->setCCSId(        rset->getInt(7) );
    result->setDCCId(        rset->getInt(8) );
    result->setLaserId(      rset->getInt(9) );
    result->setLTCId(        rset->getInt(10) );
    result->setLTSId(        rset->getInt(11) );
    result->setTCCId(        rset->getInt(12) );
    result->setTTCCIId(        rset->getInt(13) );

   
    result->setJBH4Id(        rset->getInt(14) );
    result->setScanId(        rset->getInt(15) );

    result->setTTCFId(        rset->getInt(16) );
    result->setSRPId(        rset->getInt(17) );
    result->setDCUId(        rset->getInt(18) );
    result->setTCCEEId(      rset->getInt(19) );

  } catch (SQLException &e) {
    throw(std::runtime_error("ODEcalCycle::fetchData():  "+e.getMessage()));
  }
}


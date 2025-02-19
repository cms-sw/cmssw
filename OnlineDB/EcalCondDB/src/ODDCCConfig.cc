#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cstdlib>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODDCCConfig.h"

using namespace std;
using namespace oracle::occi;

ODDCCConfig::ODDCCConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_size=0;
   m_config_tag="";
   m_ID=0;
   m_wei="";
   clear();
   
}



ODDCCConfig::~ODDCCConfig()
{
}

int ODDCCConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_dcc_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCConfig::fetchNextId():  "+e.getMessage()));
  }

}




void ODDCCConfig::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_DCC_CONFIGURATION (dcc_configuration_id, dcc_tag, "
			" DCC_CONFIGURATION_URL, TESTPATTERN_FILE_URL, "
			" N_TESTPATTERNS_TO_LOAD , SM_HALF, weightsmode, " 
			" dcc_configuration) "
                        "VALUES (:1, :2, :3, :4, :5, :6 , :7 ,:8 )");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, getConfigTag());
    m_writeStmt->setString(3, getDCCConfigurationUrl());
    m_writeStmt->setString(4, getTestPatternFileUrl());
    m_writeStmt->setInt(5, getNTestPatternsToLoad());
    m_writeStmt->setInt(6, getSMHalf());
    m_writeStmt->setString(7, getDCCWeightsMode());

    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(8,clob);
    m_writeStmt->executeUpdate ();
    m_ID=next_id; 

    m_conn->terminateStatement(m_writeStmt);
    std::cout<<"DCC Clob inserted into CONFIGURATION with id="<<next_id<<std::endl;

    // now we read and update it 
    m_writeStmt = m_conn->createStatement(); 
    m_writeStmt->setSQL ("SELECT dcc_configuration FROM ECAL_DCC_CONFIGURATION WHERE"
			 " dcc_configuration_id=:1 FOR UPDATE");

    std::cout<<"updating the clob 0"<<std::endl;
   
    
    
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCConfig::prepareWrite():  "+e.getMessage()));
  }

  std::cout<<"updating the clob 1 "<<std::endl;
  
  
}


void ODDCCConfig::setParameters(std::map<string,string> my_keys_map){

  // parses the result of the XML parser that is a map of
  // string string with variable name variable value

  for( std::map<std::string, std::string >::iterator ci=
         my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {

    if(ci->first==  "DCC_CONFIGURATION_ID") setConfigTag(ci->second);
    if(ci->first==  "TESTPATTERN_FILE_URL")   setTestPatternFileUrl(ci->second );
    if(ci->first==  "N_TESTPATTERNS_TO_LOAD") setNTestPatternsToLoad(atoi(ci->second.c_str() ));
    if(ci->first==  "SM_HALF")                setSMHalf(atoi(ci->second.c_str() ));
    if(ci->first==  "WEIGHTSMODE")           setDCCWeightsMode(ci->second.c_str() );
    if(ci->first==  "DCC_CONFIGURATION_URL") {
      std::string fname=ci->second ;
      setDCCConfigurationUrl(fname );

      // here we must open the file and read the DCC Clob
      std::cout << "Going to read DCC file: " << fname << endl;

      ifstream inpFile;
      inpFile.open(fname.c_str());

      // tell me size of file 
      int bufsize = 0; 
      inpFile.seekg( 0,ios::end ); 
      bufsize = inpFile.tellg(); 
      std::cout <<" bufsize ="<<bufsize<< std::endl;
      // set file pointer to start again 
      inpFile.seekg( 0,ios::beg ); 

      m_size=bufsize;

      inpFile.close();

    }
  }

}




void ODDCCConfig::writeDB()
  throw(std::runtime_error)
{


  std::cout<<"updating the clob "<<std::endl;


  try {

 
    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();

    rset->next ();
    oracle::occi::Clob clob = rset->getClob (1);

    cout << "Opening the clob in read write mode" << endl;

    std::cout << "Populating the clob" << endl;
    
    populateClob (clob, getDCCConfigurationUrl(), m_size);
    int clobLength=clob.length ();
    cout << "Length of the clob is: " << clobLength << endl;
  
    m_writeStmt->executeUpdate();

    m_writeStmt->closeResultSet (rset);

  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODDCCConfig::writeDB:  Failed to write"));
  }
  
  
}


void ODDCCConfig::clear(){

   m_dcc_url="";
   m_test_url="";
   m_ntest=0;
   m_sm_half=0;
   m_wei="";

}




void ODDCCConfig::fetchData(ODDCCConfig * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  //  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    //    throw(std::runtime_error("ODDCCConfig::fetchData(): no Id defined for this ODDCCConfig "));
    result->fetchID();
  }

  try {

    m_readStmt->setSQL("SELECT * "
		       "FROM ECAL_DCC_CONFIGURATION  "
		       " where  dcc_configuration_id = :1 or dcc_tag=:2 " );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    // 1 is the id and 2 is the config tag

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    result->setDCCConfigurationUrl(rset->getString(3));
    result->setTestPatternFileUrl(rset->getString(4));
    result->setNTestPatternsToLoad(rset->getInt(5));
    result->setSMHalf(rset->getInt(6));

    Clob clob = rset->getClob (7);
    m_size = clob.length();
    Stream *instream = clob.getStream (1,0);
    unsigned char *buffer = new unsigned char[m_size];
    memset (buffer, 0, m_size);
    instream->readBuffer ((char*)buffer, m_size);
    /*
    cout << "Opening the clob in Read only mode" << endl;
    clob.open (OCCI_LOB_READONLY);
    int clobLength=clob.length ();
    cout << "Length of the clob is: " << clobLength << endl;
    m_size=clobLength;
    unsigned char* buffer = readClob (clob, m_size);
    clob.close ();
    cout<< "the clob buffer is:"<<endl;  
    for (int i = 0; i < clobLength; ++i)
      cout << (char) buffer[i];
    cout << endl;


    */
    result->setDCCClob(buffer );
    result->setDCCWeightsMode(rset->getString(8));


  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCConfig::fetchData():  "+e.getMessage()));
  }
}



int ODDCCConfig::fetchID()    throw(std::runtime_error)
{
  if (m_ID!=0) {
    return m_ID;
  }
  
  this->checkConnection();
  
  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT DCC_configuration_id FROM ecal_dcc_configuration "
                 "WHERE  dcc_tag=:dcc_tag "
		 );
    
    stmt->setString(1, getConfigTag() );
    
    
    ResultSet* rset = stmt->executeQuery();
    
    if (rset->next()) {
      m_ID = rset->getInt(1);
    } else {
      m_ID = 0;
    }
    m_conn->terminateStatement(stmt);
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDCCConfig::fetchID:  "+e.getMessage()));
  }
  
  
  return m_ID;
}

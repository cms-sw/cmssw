#include <cstdlib>
#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include <algorithm>
#include <cctype>

#include "OnlineDB/EcalCondDB/interface/ODSRPConfig.h"

using namespace std;
using namespace oracle::occi;

ODSRPConfig::ODSRPConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";

   m_ID=0;
   clear();
   m_size=0;
}

void ODSRPConfig::clear(){
  //  strcpy((char *)m_srp_clob, "");
 m_debug=0;
 m_dummy=0;
 m_file="";
 m_patdir="";
 m_auto=0;
 m_bnch=0;

}

ODSRPConfig::~ODSRPConfig()
{
}

int ODSRPConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();

    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_srp_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(std::runtime_error("ODSRPConfig::fetchNextId():  "+e.getMessage()));
  }

}




void ODSRPConfig::setParameters(std::map<string,string> my_keys_map){

  // parses the result of the XML parser that is a map of
  // string string with variable name variable value


  for( std::map<std::string, std::string >::iterator ci=
         my_keys_map.begin(); ci!=my_keys_map.end(); ci++ ) {

      std::string name = ci->first;
      std::transform(name.begin(), name.end(), name.begin(), (int(*)(int))std::toupper);

    if( name ==  "SRP_CONFIGURATION_ID") setConfigTag(ci->second);
    if( name ==  "DEBUGMODE") setDebugMode(atoi(ci->second.c_str()));
    if( name ==  "DUMMYMODE") setDummyMode(atoi(ci->second.c_str()));
    if( name ==  "PATTERNDIRECTORY") setPatternDirectory(ci->second);
    if( name ==  "PATTERN_DIRECTORY") setPatternDirectory(ci->second);
    if( name ==  "AUTOMATICMASKS") setAutomaticMasks(atoi(ci->second.c_str()));
    if( name ==  "AUTOMATIC_MASKS") setAutomaticMasks(atoi(ci->second.c_str()));
    if( name ==  "AUTOMATICSRPSELECT") setAutomaticSrpSelect(atoi(ci->second.c_str()));
    if( name ==  "SRP0BUNCHADJUSTPOSITION") setSRP0BunchAdjustPosition(atoi(ci->second.c_str()));
    if( name ==  "SRP_CONFIG_FILE") {
      std::string fname=ci->second ;
    
      cout << "fname="<<fname << endl;
      setConfigFile(fname);


      // here we must open the file and read the LTC Clob
      std::cout << "Going to read SRP file: " << fname << endl;

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

void ODSRPConfig::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_SRP_CONFIGURATION (srp_configuration_id, srp_tag, "
			"   DEBUGMODE, DUMMYMODE, PATTERN_DIRECTORY, AUTOMATIC_MASKS,"
			" SRP0BUNCHADJUSTPOSITION, SRP_CONFIG_FILE, SRP_CONFIGURATION,  AUTOMATICSRPSELECT  ) "
                        "VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10 )");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, getConfigTag());
    m_writeStmt->setInt(3, getDebugMode());
    m_writeStmt->setInt(4, getDummyMode());
    m_writeStmt->setString(5, getPatternDirectory());
    m_writeStmt->setInt(6, getAutomaticMasks());
    m_writeStmt->setInt(10, getAutomaticSrpSelect());
    m_writeStmt->setInt(7, getSRP0BunchAdjustPosition());
    m_writeStmt->setString(8, getConfigFile());
  
    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(9,clob);
    m_writeStmt->executeUpdate ();
    m_ID=next_id; 

    m_conn->terminateStatement(m_writeStmt);
    std::cout<<"SRP Clob inserted into CONFIGURATION with id="<<next_id<<std::endl;

    // now we read and update it 
    m_writeStmt = m_conn->createStatement(); 
    m_writeStmt->setSQL ("SELECT srp_configuration FROM ECAL_SRP_CONFIGURATION WHERE"
			 " srp_configuration_id=:1 FOR UPDATE");

    std::cout<<"updating the clob 0"<<std::endl;

    
  } catch (SQLException &e) {
    throw(std::runtime_error("ODSRPConfig::prepareWrite():  "+e.getMessage()));
  }

  std::cout<<"updating the clob 1 "<<std::endl;
  
}


void ODSRPConfig::writeDB()
  throw(std::runtime_error)
{

  std::cout<<"updating the clob 2"<<std::endl;

  try {
    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();

    while (rset->next ())
      {
        oracle::occi::Clob clob = rset->getClob (1);
        cout << "Opening the clob in read write mode" << endl;
	cout << "Populating the clob" << endl;
	populateClob (clob, getConfigFile(), m_size );
        int clobLength=clob.length ();
        cout << "Length of the clob after writing is: " << clobLength << endl;
      
      }

    m_writeStmt->executeUpdate();

    m_writeStmt->closeResultSet (rset);

  } catch (SQLException &e) {
    throw(std::runtime_error("ODSRPConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(std::runtime_error("ODSRPConfig::writeDB:  Failed to write"));
  }


}


void ODSRPConfig::fetchData(ODSRPConfig * result)
  throw(std::runtime_error)
{
  this->checkConnection();
  //  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    //    throw(std::runtime_error("ODSRPConfig::fetchData(): no Id defined for this ODSRPConfig "));
    result->fetchID();
  }

  try {

    m_readStmt->setSQL("SELECT  * "
		       " FROM ECAL_SRP_CONFIGURATION  "
		       " where (srp_configuration_id = :1 or srp_tag=:2 )" );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();
    // 1 is the id and 2 is the config tag

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));

    result->setDebugMode(rset->getInt(3));
    result->setDummyMode(rset->getInt(4));
    result->setPatternDirectory(rset->getString(5));
    result->setAutomaticMasks(rset->getInt(6));
    result->setSRP0BunchAdjustPosition(rset->getInt(7));
    result->setConfigFile(rset->getString(8));

    Clob clob = rset->getClob(9);
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
    unsigned char* buffer = readClob (clob, clobLength);
    clob.close ();
    cout<< "the clob buffer is:"<<endl;  
    for (int i = 0; i < clobLength; ++i)
      cout << (char) buffer[i];
    cout << endl;


    */
    result->setSRPClob(buffer );
    result->setAutomaticSrpSelect(rset->getInt(10));

  } catch (SQLException &e) {
    throw(std::runtime_error("ODSRPConfig::fetchData():  "+e.getMessage()));
  }
}



int ODSRPConfig::fetchID()    throw(std::runtime_error)
{
  // Return from memory if available
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT srp_configuration_id FROM ecal_srp_configuration "
                 "WHERE  srp_tag=:srp_tag "
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
    throw(std::runtime_error("ODSRPConfig::fetchID:  "+e.getMessage()));
  }

    return m_ID;
}

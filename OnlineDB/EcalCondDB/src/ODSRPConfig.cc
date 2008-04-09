#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

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

}

void ODSRPConfig::clear(){
  strcpy((char *)m_srp_clob, "");
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
    throw(runtime_error("ODSRPConfig::fetchNextId():  "+e.getMessage()));
  }

}




void ODSRPConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_SRP_CONFIGURATION (srp_configuration_id, srp_tag, "
			"   DEBUGMODE, DUMMYMODE, PATTERN_DIRECTORY, AUTOMATIC_MASKS, "
			" SRP0BUNCHADJUSTPOSITION, SRP_CONFIG_FILE, SRP_CONFIGURATION  ) "
                        "VALUES (:1, :2, :3, :4, :5, :6, :7, :8 )");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, getConfigTag());
    m_writeStmt->setInt(3, getDebugMode());
    m_writeStmt->setInt(4, getDummyMode());
    m_writeStmt->setString(5, getPatternDirectory());
    m_writeStmt->setInt(6, getAutomaticMasks());
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
    throw(runtime_error("ODSRPConfig::prepareWrite():  "+e.getMessage()));
  }

  std::cout<<"updating the clob 1 "<<std::endl;
  
}
//
void ODSRPConfig::dumpClob (oracle::occi::Clob &clob,unsigned int way)
   throw (std::runtime_error)
  {

  try{
    unsigned int size=BUFSIZE;
    unsigned int offset = 1;
  
    if (clob.isNull())
    {
       cout << "Clob is Null\n";
       return;
    }
    unsigned int cloblen = clob.length();
    cout << "Length of Clob : "<< cloblen << endl;
    if (cloblen == 0)
       return;
    unsigned char *buffer= new unsigned char[size]; 
    memset (buffer, NULL, size);
    if (way==USE_NORM)
    {
       cout << "Dumping clob (using read ): ";
       int bytesRead=clob.read(size,buffer,size,offset);
       for (int i = 0; i < bytesRead; ++i)
          cout << buffer[i];
       cout << endl;
    }
    else if(way==USE_BUFF)
    {
       Stream *inStream = clob.getStream (1,0);
       cout << "Dumping clob(using stream): ";
       int bytesRead=(inStream->readBuffer((char *)buffer, size));
       while (bytesRead > 0)
       {
          for (int i = 0; i < bytesRead; ++i) 
          {
              cout << buffer[i];
          }
          bytesRead=(inStream->readBuffer((char *)buffer, size));
       }
       cout << endl;
       clob.closeStream (inStream);
    }
    delete []buffer;
  } catch (SQLException &e) {
    throw(runtime_error("ODSRPConfig::prepareWrite():  "+e.getMessage()));
  }

}


void ODSRPConfig::writeDB()
  throw(runtime_error)
{


  std::cout<<"updating the clob 2"<<std::endl;


  try {


    m_writeStmt->setInt(1, m_ID);
    ResultSet* rset = m_writeStmt->executeQuery();

    while (rset->next ())
      {
        oracle::occi::Clob clob = rset->getClob (1);
        cout << "Opening the clob in read write mode" << endl;

        cout << "dumping the clob" << endl;
	dumpClob (clob, USE_NORM);
	cout << "Populating the clob" << endl;
	populateClob (clob);
        int clobLength=clob.length ();
        cout << "Length of the clob is: " << clobLength << endl;
        clob.close ();
      }

    m_writeStmt->executeUpdate();

    m_writeStmt->closeResultSet (rset);

  } catch (SQLException &e) {
    throw(runtime_error("ODSRPConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODSRPConfig::writeDB:  Failed to write"));
  }


}



unsigned char* ODSRPConfig::readClob (oracle::occi::Clob &clob, int size)
  throw (runtime_error)
{

  try{
    Stream *instream = clob.getStream (1,0);
    unsigned char *buffer= new unsigned char[size]; 
    memset (buffer, NULL, size);
    
    instream->readBuffer ((char*)buffer, size);
    cout << "remember to delete the char* at the end of the program ";
       for (int i = 0; i < size; ++i)
       cout << (char) buffer[i];
     cout << endl;
    

    clob.closeStream (instream);

    return  buffer;

  }catch (SQLException &e) {
    throw(runtime_error("ODSRPConfig::dumpClob():  "+e.getMessage()));
  }

}

/**
 * populating the clob;
 */
void ODSRPConfig::populateClob (oracle::occi::Clob &clob)
  throw (std::runtime_error)
{

  if (clob.isNull())
    {
      cout << "Clob is Null\n";
      return;
    }

  try{
    
    unsigned int offset=1;
    unsigned int  my_size= strlen((char*)m_srp_clob);
    std::cout<<" size is"<< my_size<< std::endl;  
    std::cout<<" m_srp_clob is"<< m_srp_clob<< std::endl;  
    
    clob.open(OCCI_LOB_READWRITE);
    unsigned int bytesWritten=clob.write (my_size,m_srp_clob, my_size,offset);
    
  }catch (SQLException &e) {
    throw(runtime_error("ODSRPConfig::populateClob():  "+e.getMessage()));
  }
  
}




void ODSRPConfig::fetchData(ODSRPConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();
  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(runtime_error("ODSRPConfig::fetchData(): no Id defined for this ODSRPConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT  d.srp_configuration_id, d.srp_tag, "
		       " d.DEBUGMODE, d.DUMMYMODE, d.PATTERN_DIRECTORY, d.AUTOMATIC_MASKS,"
		       " d.SRP0BUNCHADJUSTPOSITION, d.SRP_CONFIG_FILE, d.SRP_CONFIGURATION "
		       " FROM ECAL_SRP_CONFIGURATION d "
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


    result->setSRPClob(buffer );

  } catch (SQLException &e) {
    throw(runtime_error("ODSRPConfig::fetchData():  "+e.getMessage()));
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
    throw(runtime_error("ODSRPConfig::fetchID:  "+e.getMessage()));
  }

    return m_ID;
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODTTCFConfig.h"

using namespace std;
using namespace oracle::occi;

ODTTCFConfig::ODTTCFConfig()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_config_tag="";
  m_ID=0;
  clear();
}


void ODTTCFConfig::clear(){
  strcpy((char *)m_ttcf_clob, "");
}


ODTTCFConfig::~ODTTCFConfig()
{
}

int ODTTCFConfig::fetchNextId()  throw(std::runtime_error) {

  int result=0;
  try {
    this->checkConnection();
    std::cout<< "going to fetch new id for TTCF 1"<<endl;
    m_readStmt = m_conn->createStatement(); 
    m_readStmt->setSQL("select ecal_ttcf_config_sq.NextVal from dual");
    ResultSet* rset = m_readStmt->executeQuery();
    while (rset->next ()){
      result= rset->getInt(1);
    }
    std::cout<< "id is : "<< result<<endl;

    m_conn->terminateStatement(m_readStmt);
    return result; 

  } catch (SQLException &e) {
    throw(runtime_error("ODTTCFConfig::fetchNextId():  "+e.getMessage()));
  }

}




void ODTTCFConfig::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();
  int next_id=fetchNextId();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO ECAL_TTCF_CONFIGURATION (ttcf_configuration_id, ttcf_tag, " 
			" configuration ) "
                        "VALUES (:1, :2, :3 )");
    m_writeStmt->setInt(1, next_id);
    m_writeStmt->setString(2, getConfigTag());
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(3,clob);
    m_writeStmt->executeUpdate ();
    m_ID=next_id; 

    m_conn->terminateStatement(m_writeStmt);
    std::cout<<"inserted into CONFIGURATION with id="<<next_id<<std::endl;

    // now we read and update it 
    m_writeStmt = m_conn->createStatement(); 
    m_writeStmt->setSQL ("SELECT configuration FROM ECAL_TTCF_CONFIGURATION WHERE"
			 " ttcf_configuration_id=:1 FOR UPDATE");

  std::cout<<"updating the clob 0"<<std::endl;


  } catch (SQLException &e) {
    throw(runtime_error("ODTTCFConfig::prepareWrite():  "+e.getMessage()));
  }

  std::cout<<"updating the clob 1 "<<std::endl;

}
//
void ODTTCFConfig::dumpClob (oracle::occi::Clob &clob,unsigned int way)
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
    throw(runtime_error("ODTTCFConfig::prepareWrite():  "+e.getMessage()));
  }

}


void ODTTCFConfig::writeDB()
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
    throw(runtime_error("ODTTCFConfig::writeDB():  "+e.getMessage()));
  }
  // Now get the ID
  if (!this->fetchID()) {
    throw(runtime_error("ODTTCFConfig::writeDB:  Failed to write"));
  }


}



char* ODTTCFConfig::readClob (oracle::occi::Clob &clob, int size)
  throw (runtime_error)
{

  try{
    Stream *instream = clob.getStream (1,0);
    char *buffer = new char[size];
    memset (buffer, NULL, size);
    
    instream->readBuffer (buffer, size);
    cout << "remember to delete the char* at the end of the program ";
       for (int i = 0; i < size; ++i)
       cout << (char) buffer[i];
     cout << endl;
    

    clob.closeStream (instream);

    return buffer;

  }catch (SQLException &e) {
    throw(runtime_error("ODTTCFConfig::dumpClob():  "+e.getMessage()));
  }

}

/**
 * populating the clob;
 */
void ODTTCFConfig::populateClob (oracle::occi::Clob &clob)
  throw (std::runtime_error)
{

  if (clob.isNull())
    {
      cout << "Clob is Null\n";
      return;
    }

  try{
    
    unsigned int offset=1;
    unsigned int  my_size= strlen((char*)m_ttcf_clob);
    std::cout<<" size is"<< my_size<< std::endl;  
    std::cout<<" m_ttcf_clob is"<< m_ttcf_clob<< std::endl;  
    
    clob.open(OCCI_LOB_READWRITE);
    unsigned int bytesWritten=clob.write (my_size,m_ttcf_clob, my_size,offset);
    
  }catch (SQLException &e) {
    throw(runtime_error("ODTTCFConfig::populateClob():  "+e.getMessage()));
  }
  
}




void ODTTCFConfig::fetchData(ODTTCFConfig * result)
  throw(runtime_error)
{
  this->checkConnection();
  result->clear();

  if(result->getId()==0 && (result->getConfigTag()=="") ){
    throw(runtime_error("ODTTCFConfig::fetchData(): no Id defined for this ODTTCFConfig "));
  }

  try {

    m_readStmt->setSQL("SELECT *   "
		       "FROM ECAL_TTCF_CONFIGURATION  "
		       " where (ttcf_configuration_id = :1 or ttcf_tag= :2) " );
    m_readStmt->setInt(1, result->getId());
    m_readStmt->setString(2, result->getConfigTag());
    ResultSet* rset = m_readStmt->executeQuery();

    rset->next();

    result->setId(rset->getInt(1));
    result->setConfigTag(rset->getString(2));
    Clob clob = rset->getClob (3);
    cout << "Opening the clob in Read only mode" << endl;
    clob.open (OCCI_LOB_READONLY);
    int clobLength=clob.length ();
    cout << "Length of the clob is: " << clobLength << endl;
    char* buffer = readClob (clob, clobLength);
    clob.close ();
    result->setTTCFClob((unsigned char*) buffer );

  } catch (SQLException &e) {
    throw(runtime_error("ODTTCFConfig::fetchData():  "+e.getMessage()));
  }
}



int ODTTCFConfig::fetchID()    throw(std::runtime_error)
{
  if (m_ID!=0) {
    return m_ID;
  }

  this->checkConnection();

  try {
    Statement* stmt = m_conn->createStatement();
    stmt->setSQL("SELECT ttcf_configuration_id FROM ecal_ttcf_configuration "
                 "WHERE  ttcf_tag=:ttcf_tag "
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
    throw(runtime_error("ODTTCFConfig::fetchID:  "+e.getMessage()));
  }

    return m_ID;
}

#ifndef IODCONFIG_H
#define IODCONFIG_H
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cstring>


#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

/**
 *   Abstract interface for data in the conditions DB
 */

class IODConfig : public IDBObject {
  
 public:
  typedef oracle::occi::SQLException SQLException;
  typedef oracle::occi::Statement Statement;
  typedef oracle::occi::Stream Stream;
  typedef oracle::occi::Clob Clob;
  
  std::string   m_config_tag;
  
  virtual std::string getTable() =0;
  
  IODConfig():IDBObject() {
    m_debug = 0;
  };
    
    inline void setDebug() {
      m_debug = 1;
      std::cout << "------ DEBUG on" << std::endl << std::flush;
    }
    inline int getDebugLevel() {
      return m_debug;
    }
    inline void noDebug() {
      m_debug = 0;
    }
    inline void setConfigTag(std::string x) {m_config_tag=x;}
    inline std::string getConfigTag() {return m_config_tag;}
    
    
 protected:
    Statement* m_writeStmt;
    Statement* m_readStmt;
    int m_debug;
    
    inline void checkPrepare() 
      throw(std::runtime_error) 
      {
	if (m_writeStmt == NULL) {
	  throw(std::runtime_error("Write statement not prepared"));
	}
      }
    
    inline void terminateWriteStatement()
      throw(std::runtime_error)
      {
	if (m_writeStmt != NULL) {
	  m_conn->terminateStatement(m_writeStmt);
	} else {
	  std::cout << "Warning from IDataItem: statement was aleady closed"<< std::endl;
	}
      }
    
    
    inline void createReadStatement()
      throw(std::runtime_error)
      {
	m_readStmt=m_conn->createStatement();
      }
    
    inline void setPrefetchRowCount(int ncount)
      throw(std::runtime_error)
      {
	m_readStmt->setPrefetchRowCount(ncount);
      }
    
    inline void terminateReadStatement()
      throw(std::runtime_error)
      {
	if (m_readStmt != NULL) {
	  m_conn->terminateStatement(m_readStmt);
	} else {
	  std::cout << "Warning from IDataItem: statement was aleady closed"<< std::endl;
	}
      }
    
    
    
    // Prepare a statement for writing operations
    virtual void prepareWrite() throw(std::runtime_error) =0;
    
    //  virtual void writeDB() throw(std::runtime_error) ;


void populateClob (Clob &clob, std::string fname, unsigned int bufsize)
 throw (std::runtime_error)
{

  try{
    // Uses stream here
    if (m_debug) {
      std::cout << "Populating the Clob using writeBuffer(Stream) method" << std::endl;
      std::cout<<"we are here0"<<std::endl
	       << std::flush; 
    }
      
    char *file = (char *)fname.c_str();
    if (m_debug) {
      std::cout<<"we are here0.5 file is:"<<fname<<std::endl
	       << std::flush; 
    }
    
    std::ifstream inFile;
    inFile.open(file,std::ios::in);
    if (!inFile)
      {
	std::cout << fname <<" file not found\n";
	inFile.close();
	
	std::string fname2="/nfshome0/ecaldev/francesca/null_file.txt";
	inFile.open((char*)fname2.c_str(),std::ios::in);
      }
    if(bufsize==0){
      
      
      inFile.seekg( 0,std::ios::end ); 
      bufsize = inFile.tellg(); 
      if (m_debug) {
	std::cout <<" bufsize ="<<bufsize<< std::endl 
		  << std::flush;
      }
      // set file pointer to start again 
      inFile.seekg( 0,std::ios::beg ); 
      
    }
    
    char *buffer = new char[bufsize + 1];
    
    if (m_debug) {
      std::cout<<"we are here1"<<std::endl << std::flush; 
    }
    unsigned int size;
    Stream *strm=clob.getStream();
    if (m_debug) {
      std::cout<<"we are here2"<<std::endl << std::flush; 
    }
    //    while(inFile)
    //	{
    int buf=0;
    memset (buffer, buf, bufsize + 1);
    inFile.read(buffer,bufsize);
    if (m_debug) {
      std::cout<<"we are here2.5"<<std::endl << std::flush; 
    }
    strm->writeBuffer(buffer,strlen(buffer));
    if (m_debug) {
      std::cout<<"we are here2.6"<<std::endl << std::flush; 
      std::cout<<"we are here3"<<std::endl << std::flush; 
    }
    
    //}
    strcpy(buffer," ");
    size=strlen(buffer);
    strm->writeLastBuffer(buffer,size);
    clob.closeStream(strm);
    inFile.close();
    if (m_debug) {
      std::cout<<"we are here4"<<std::endl << std::flush; 
    }
    delete[] buffer;
    
    
  }catch (SQLException &e) {
    throw(std::runtime_error("populateClob():  "+e.getMessage()));
  }
  if (m_debug) {
    std::cout << "Populating the Clob - Success" << std::endl
	      << std::flush;
  }
}


unsigned char* readClob (Clob &clob, int size)
  throw (std::runtime_error)
{
  // TO BE REMOVED!!!!!!!!
  unsigned char *buffer = new unsigned char[size];
  return readClob(clob, size, buffer);
}

unsigned char* readClob (Clob &clob, int size, unsigned char *buffer)
  throw (std::runtime_error)
{

  try{
    Stream *instream = clob.getStream (1,0);
    buffer= new unsigned char[size]; 
    int buf=0;
    memset (buffer, buf, size);
    
    instream->readBuffer ((char*)buffer, size);
    //std::cout << "remember to delete the char* at the end of the program ";
    //  for (int i = 0; i < size; ++i)
    // std::cout << (char) buffer[i];
    // std::cout << std::endl;
    

    clob.closeStream (instream);

    return  buffer;

  }catch (SQLException &e) {
    throw(std::runtime_error("readClob():  "+e.getMessage()));
  }

}



};

#endif



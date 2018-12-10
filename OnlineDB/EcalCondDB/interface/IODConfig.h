#ifndef IODCONFIG_H
#define IODCONFIG_H
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>
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

  inline void setConfigTag(std::string x) {m_config_tag=x;}
  inline std::string getConfigTag() {return m_config_tag;}
 

 protected:
  Statement* m_writeStmt;
  Statement* m_readStmt;

  inline void checkPrepare() noexcept(false)
    {
      if (m_writeStmt == nullptr) {
	throw(std::runtime_error("Write statement not prepared"));
      }
    }

  inline void terminateWriteStatement() noexcept(false)
  {
    if (m_writeStmt != nullptr) {
      m_conn->terminateStatement(m_writeStmt);
    } else {
      std::cout << "Warning from IDataItem: statement was aleady closed"<< std::endl;
    }
  }


  inline void createReadStatement() noexcept(false)
  {
      m_readStmt=m_conn->createStatement();
  }

  inline void setPrefetchRowCount(int ncount) noexcept(false)
  {
    m_readStmt->setPrefetchRowCount(ncount);
  }

  inline void terminateReadStatement() noexcept(false)
  {
    if (m_readStmt != nullptr) {
      m_conn->terminateStatement(m_readStmt);
    } else {
      std::cout << "Warning from IDataItem: statement was aleady closed"<< std::endl;
    }
  }



  // Prepare a statement for writing operations
  virtual void prepareWrite() noexcept(false) = 0;

  //  virtual void writeDB() noexcept(false) ;


void populateClob (Clob &clob, std::string fname, unsigned int bufsize) noexcept(false)
{

  try{
      // Uses stream here
      std::cout << "Populating the Clob using writeBuffer(Stream) method" << std::endl;
      std::cout<<"we are here0"<<std::endl; 

      const char *file = fname.c_str();
      std::cout<<"we are here0.5 file is:"<<fname<<std::endl; 

      std::ifstream inFile;
      inFile.open(file,std::ios::in);
      if (!inFile)
	{
          std::cout << fname <<" file not found\n";
	  inFile.close();

	  std::string fname2="/nfshome0/ecaldev/francesca/null_file.txt";
	  inFile.open(fname2.c_str(),std::ios::in);
	  

          
	}
      if(bufsize==0){


	inFile.seekg( 0,std::ios::end ); 
	bufsize = inFile.tellg(); 
	std::cout <<" bufsize ="<<bufsize<< std::endl;
	// set file pointer to start again 
	inFile.seekg( 0,std::ios::beg ); 
	
      }

      char *buffer = new char[bufsize + 1];


      std::cout<<"we are here1"<<std::endl; 
      unsigned int size;
      Stream *strm=clob.getStream();
      std::cout<<"we are here2"<<std::endl; 
      //    while(inFile)
      //	{
      int buf=0;
      memset (buffer, buf, bufsize + 1);
      inFile.read(buffer,bufsize);
      std::cout<<"we are here2.5"<<std::endl; 
      
      strm->writeBuffer(buffer,strlen(buffer));
      std::cout<<"we are here2.6"<<std::endl; 

      //}
      std::cout<<"we are here3"<<std::endl; 
      strcpy(buffer," ");
      size=strlen(buffer);
      strm->writeLastBuffer(buffer,size);
      clob.closeStream(strm);
      inFile.close();
      std::cout<<"we are here4"<<std::endl; 
      delete[] buffer;


  }catch (SQLException &e) {
    throw(std::runtime_error(std::string("populateClob():  ")+getOraMessage(&e)));
  }

  std::cout << "Populating the Clob - Success" << std::endl;
}


unsigned char* readClob (Clob &clob, int size) noexcept(false)
{

  try{
    Stream *instream = clob.getStream (1,0);
    unsigned char *buffer= new unsigned char[size]; 
    int buf=0;
    memset (buffer, buf, size);
    
    instream->readBuffer ((char*)buffer, size);
    std::cout << "remember to delete the char* at the end of the program ";
       for (int i = 0; i < size; ++i)
       std::cout << (char) buffer[i];
     std::cout << std::endl;
    

    clob.closeStream (instream);

    return  buffer;

  }catch (SQLException &e) {
    throw(std::runtime_error(std::string("readClob():  ")+getOraMessage(&e)));
  }

}



};

#endif



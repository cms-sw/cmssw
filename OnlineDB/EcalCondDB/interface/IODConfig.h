#ifndef IODCONFIG_H
#define IODCONFIG_H
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>


#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"


using namespace std;
using namespace oracle::occi;


/**
 *   Abstract interface for data in the conditions DB
 */

class IODConfig : public IDBObject {

 public:

  std::string   m_config_tag;

  virtual std::string getTable() =0;

  inline void setConfigTag(std::string x) {m_config_tag=x;}
  inline std::string getConfigTag() {return m_config_tag;}
 

 protected:
  oracle::occi::Statement* m_writeStmt;
  oracle::occi::Statement* m_readStmt;

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
      cout << "Warning from IDataItem: statement was aleady closed"<< endl;
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
      cout << "Warning from IDataItem: statement was aleady closed"<< endl;
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
      cout << "Populating the Clob using writeBuffer(Stream) method" << endl;
      std::cout<<"we are here0"<<std::endl; 

      char *file = (char *)fname.c_str();
      std::cout<<"we are here0.5 file is:"<<fname<<std::endl; 

      ifstream inFile;
      inFile.open(file,ios::in);
      if (!inFile)
	{
          cout << fname <<" file not found\n";
	  inFile.close();

	  std::string fname2="/nfshome0/ecaldev/francesca/null_file.txt";
	  inFile.open((char*)fname2.c_str(),ios::in);
	  

          
	}
      if(bufsize==0 || bufsize==-1){


	inFile.seekg( 0,ios::end ); 
	bufsize = inFile.tellg(); 
	std::cout <<" bufsize ="<<bufsize<< std::endl;
	// set file pointer to start again 
	inFile.seekg( 0,ios::beg ); 
	
      }

      char *buffer = new char[bufsize + 1];


      std::cout<<"we are here1"<<std::endl; 
      unsigned int size;
      Stream *strm=clob.getStream();
      std::cout<<"we are here2"<<std::endl; 
      //    while(inFile)
      //	{
      memset (buffer, NULL, bufsize + 1);
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
    throw(runtime_error("populateClob():  "+e.getMessage()));
  }

  cout << "Populating the Clob - Success" << endl;
}


unsigned char* readClob (oracle::occi::Clob &clob, int size)
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
    throw(runtime_error("readClob():  "+e.getMessage()));
  }

}



};

#endif



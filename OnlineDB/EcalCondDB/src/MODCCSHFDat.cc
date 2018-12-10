#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>

#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MODCCSHFDat.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"


using namespace std;
using namespace oracle::occi;

MODCCSHFDat::MODCCSHFDat()
{
  m_env = nullptr;
  m_conn = nullptr;
  m_writeStmt = nullptr;
  m_readStmt = nullptr;

  //  m_clob = 0;
  m_size=0;
  m_file="";

}



MODCCSHFDat::~MODCCSHFDat()
{
}


void MODCCSHFDat::setFile(std::string x) {
  m_file=x;
  //try {
  std::cout<< "file is "<< m_file<<endl;
  // }catch (Exception &e) {
  //throw(std::runtime_error(std::string("MODCCSHFDat::setFile():  ")+getOraMessage(&e)));
  //} 
    // here we must open the file and read the CCS Clob
    std::cout << "Going to read CCS file: " << m_file << endl;
    ifstream inpFile;
    inpFile.open(m_file.c_str());
    // tell me size of file
    int bufsize = 0;
    inpFile.seekg( 0,ios::end );
    bufsize = inpFile.tellg();
    std::cout <<" bufsize ="<<bufsize<< std::endl;
    // set file pointer to start again
    inpFile.seekg( 0,ios::beg );
    m_size=bufsize;
    inpFile.close();
    std::cout << "Going to read CCS file: " << m_file << endl;

}

void MODCCSHFDat::prepareWrite()
  noexcept(false)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO OD_CCS_HF_dat (iov_id, logic_id, "
			"ccs_log) "
			"VALUES (:iov_id, :logic_id, "
			":ccs_log )");


  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("MODCCSHFDat::prepareWrite():  ")+getOraMessage(&e)));
  }
}



void MODCCSHFDat::writeDB(const EcalLogicID* ecid, const MODCCSHFDat* item, MODRunIOV* iov )
  noexcept(false)
{



  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MODCCSHFDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("MODCCSHFDat::writeDB:  Bad EcalLogicID")); }
  
  std::string fname=item->getFile();
  std::cout << "Going to read CCS file: " << fname << endl;


  try {

    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    // and now the clob
    oracle::occi::Clob clob(m_conn);
    clob.setEmpty();
    m_writeStmt->setClob(3,clob);
    m_writeStmt->executeUpdate ();

    m_conn->terminateStatement(m_writeStmt);
    std::cout<<"empty CCS Clob inserted into DB" <<std::endl;

    // now we read and update it
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL ("SELECT CCS_LOG FROM OD_CCS_HF_DAT WHERE"
                         " iov_ID=:1 and logic_ID=:2 FOR UPDATE");
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    ResultSet* rset = m_writeStmt->executeQuery();
    rset->next ();
    oracle::occi::Clob clob2 = rset->getClob (1);
    cout << "Opening the clob in read write mode" << endl;

    unsigned int clob_size=item->getSize();

    populateClob (clob2 , fname, clob_size);

    int clobLength=clob2.length ();
    cout << "Length of the clob is: " << clobLength << endl;

    m_writeStmt->executeUpdate();
    m_writeStmt->closeResultSet (rset);

  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("MODCCSHFDat::writeDB():  ")+getOraMessage(&e)));
  }
}



void MODCCSHFDat::fetchData(std::map< EcalLogicID, MODCCSHFDat >* fillMap, MODRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("MODCCSHFDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 " d.ccs_log " 
		 "FROM channelview cv JOIN OD_CCS_HF_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, MODCCSHFDat > p;
    MODCCSHFDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( getOraString(rset,1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     getOraString(rset,6));    // maps_to
      // to be corrected 
      //      dat.setClob( getOraString(rset,7) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("MODCCSHFDat::fetchData():  ")+getOraMessage(&e)));
  }
}

void MODCCSHFDat::writeArrayDB(const std::map< EcalLogicID, MODCCSHFDat >* data, MODRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MODCCSHFDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  int* xx= new int[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];

  const EcalLogicID* channel;
  //const MODCCSHFDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, MODCCSHFDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("MODCCSHFDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	// dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	// to be corrected 

	int x=0;
	//=dataitem->getWord();

	xx[count]=x;

	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] xx;


    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;



  } catch (SQLException &e) {
    throw(std::runtime_error(std::string("MonPedestalsDat::writeArrayDB():  ")+getOraMessage(&e)));
  }
}



void MODCCSHFDat::populateClob (Clob &clob, std::string fname, unsigned int clob_size )
 noexcept(false)
{

  try{
      // Uses stream here
      cout << "Populating the Clob using writeBuffer(Stream) method" << endl;
      std::cout<<"we are here0"<<std::endl; 

      std::cout<<"we are here0.5 file is:"<<fname<<std::endl; 

      ifstream inFile;
      inFile.open(fname.c_str(),ios::in);
      if (!inFile)
	{
          cout << fname<<" file not found\n";
	  inFile.close();

	  std::string fname2="/u1/fra/null_file.txt";
	  inFile.open(fname2.c_str(),ios::in);
	  

          
	}
      if(clob_size==0){


	inFile.seekg( 0,ios::end ); 
	clob_size = inFile.tellg(); 
	std::cout <<" bufsize ="<<clob_size<< std::endl;
	// set file pointer to start again 
	inFile.seekg( 0,ios::beg ); 
	
      }

      char *buffer = new char[clob_size + 1];


      std::cout<<"we are here1"<<std::endl; 
      unsigned int size;
      Stream *strm=clob.getStream();
      std::cout<<"we are here2"<<std::endl; 
      //    while(inFile)
      //	{
      int buf=0;
      memset (buffer, buf, clob_size + 1);
      inFile.read(buffer,clob_size);
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

  cout << "Populating the Clob - Success" << endl;
}


unsigned char* MODCCSHFDat::readClob (oracle::occi::Clob &clob, int size)
  noexcept(false)
{

  try{
    Stream *instream = clob.getStream (1,0);
    unsigned char *buffer= new unsigned char[size]; 
    int buf=0;
    memset (buffer, buf, size);
    
    instream->readBuffer ((char*)buffer, size);
    cout << "remember to delete the char* at the end of the program ";
       for (int i = 0; i < size; ++i)
       cout << (char) buffer[i];
     cout << endl;
    

    clob.closeStream (instream);

    return  buffer;

  }catch (SQLException &e) {
    throw(std::runtime_error(std::string("readClob():  ")+getOraMessage(&e)));
  }

}

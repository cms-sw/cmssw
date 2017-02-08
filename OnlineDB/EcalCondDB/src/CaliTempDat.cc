#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/CaliTempDat.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"

using namespace std;
using namespace oracle::occi;

CaliTempDat::CaliTempDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;

  m_beta = 0;
  m_r25 = 0;
  m_offset = 0;
  m_taskStatus = false;
}



CaliTempDat::~CaliTempDat()
{
}



void CaliTempDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();
  
  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO cali_temp_dat (iov_id, logic_id, "
			" beta, r25, offset,  task_status) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5, :6 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("CaliTempDat::prepareWrite():  "+e.getMessage()));
  }
}



void CaliTempDat::writeDB(const EcalLogicID* ecid, const CaliTempDat* item, CaliIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();
  
  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("CaliTempDat::writeDB:  IOV not in DB")); }
  
  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("CaliTempDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    
    m_writeStmt->setFloat(3, item->getBeta() );
    m_writeStmt->setFloat(4, item->getR25() );
    m_writeStmt->setFloat(5, item->getOffset() );
    m_writeStmt->setInt(6, item->getTaskStatus() );
    
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("CaliTempDat::writeDB():  "+e.getMessage()));
  }
}



void CaliTempDat::fetchData(std::map< EcalLogicID, CaliTempDat >* fillMap, CaliIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();
  
  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("CaliTempDat::writeDB:  IOV not in DB")); 
    return;
  }
  
  try {
    
    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.beta, d.r25, d.offset, d.task_status "
		 "FROM channelview cv JOIN cali_temp_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, CaliTempDat > p;
    CaliTempDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to
      
      dat.setBeta( rset->getFloat(7) );
      dat.setR25( rset->getFloat(8) );
      dat.setOffset( rset->getFloat(9) );
      dat.setTaskStatus( rset->getInt(10) );
      
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("CaliTempDat::fetchData():  "+e.getMessage()));
  }
}

void CaliTempDat::writeArrayDB(const std::map< EcalLogicID, CaliTempDat >* data, CaliIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("CaliTempDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size();
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
  int* st= new int[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];

  const EcalLogicID* channel;
  const CaliTempDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, CaliTempDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
    channel = &(p->first);
    int logicID = channel->getLogicID();
    if (!logicID) { throw(std::runtime_error("CaliTempDat::writeArrayDB:  Bad EcalLogicID")); }
    ids[count]=logicID;
    iovid_vec[count]=iovID;

    dataitem = &(p->second);
    // dataIface.writeDB( channel, dataitem, iov);
    float x=dataitem->getBeta();
    float y=dataitem->getR25();
    float z=dataitem->getOffset();
    int statu=dataitem->getTaskStatus();



    xx[count]=x;
    yy[count]=y;
    zz[count]=z;
    st[count]=statu;


    ids_len[count]=sizeof(ids[count]);
    iov_len[count]=sizeof(iovid_vec[count]);

    x_len[count]=sizeof(xx[count]);
    y_len[count]=sizeof(yy[count]);
    z_len[count]=sizeof(zz[count]);
    st_len[count]=sizeof(st[count]);

    count++;
  }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );


    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] st;


    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;

    delete [] st_len;



  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalsDat::writeArrayDB():  "+e.getMessage()));
  }
}

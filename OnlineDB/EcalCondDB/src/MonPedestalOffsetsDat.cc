#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalOffsetsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPedestalOffsetsDat::MonPedestalOffsetsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_dacG1 = 0;
  m_dacG6 = 0;
  m_dacG12 = 0;
  m_taskStatus = 0;
}



MonPedestalOffsetsDat::~MonPedestalOffsetsDat()
{
}



void MonPedestalOffsetsDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_pedestal_offsets_dat (iov_id, logic_id, "
			"dac_g1, dac_g6, dac_g12, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":dac_g1, :dac_g6, :dac_g12, :task_status)");
  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalOffsetsDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonPedestalOffsetsDat::writeDB(const EcalLogicID* ecid, const MonPedestalOffsetsDat* item, MonRunIOV* iov )
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonPedestalOffsetsDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("MonPedestalOffsetsDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getDACG1() );
    m_writeStmt->setInt(4, item->getDACG6() );
    m_writeStmt->setInt(5, item->getDACG12() );
    m_writeStmt->setInt(6, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalOffsetsDat::writeDB():  "+e.getMessage()));
  }
}


void MonPedestalOffsetsDat::fetchData(std::map< EcalLogicID, MonPedestalOffsetsDat >* fillMap, MonRunIOV* iov,  std::string mappa )
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("MonPedestalOffsetsDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.dac_g1, d.dac_g6, d.dac_g12, d.task_status "
		 "FROM channelview cv JOIN mon_pedestal_offsets_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = " + mappa +
		 " WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, MonPedestalOffsetsDat > p;
    MonPedestalOffsetsDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setDACG1( rset->getInt(7) );
      dat.setDACG6( rset->getInt(8) );
      dat.setDACG12( rset->getInt(9) );
      dat.setTaskStatus( rset->getInt(10) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalOffsetsDat::fetchData():  "+e.getMessage()));
  }
}

void MonPedestalOffsetsDat::writeArrayDB(const std::map< EcalLogicID, MonPedestalOffsetsDat >* data, MonRunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonPedestalOffsetsDat::writeArrayDB:  IOV not in DB")); }


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
  const MonPedestalOffsetsDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, MonPedestalOffsetsDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("MonPedestalOffsetsDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	float x=dataitem->getDACG1();
	float y=dataitem->getDACG6();
	float z=dataitem->getDACG12();
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
    throw(std::runtime_error("MonPedestalOffsetsDat::writeArrayDB():  "+e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonPedestalsDat::MonPedestalsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_pedMeanG1 = 0;
  m_pedMeanG6 = 0;
  m_pedMeanG12 = 0;
  m_pedRMSG1 = 0;
  m_pedRMSG6 = 0;
  m_pedRMSG12 = 0;
  m_taskStatus = 0;
}



MonPedestalsDat::~MonPedestalsDat()
{
}



void MonPedestalsDat::prepareWrite()
  noexcept(false)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_pedestals_dat (iov_id, logic_id, "
		      "ped_mean_g1, ped_mean_g6, ped_mean_g12, "
		      "ped_rms_g1, ped_rms_g6, ped_rms_g12, task_status) "
		      "VALUES (:iov_id, :logic_id, "
		      ":ped_mean_g1, :ped_mean_g6, :ped_mean_g12, "
		      ":ped_rms_g1, :ped_rms_g6, :ped_rms_g12, :task_status)");
  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalsDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonPedestalsDat::writeDB(const EcalLogicID* ecid, const MonPedestalsDat* item, MonRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonPedestalsDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("MonPedestalsDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getPedMeanG1());
    m_writeStmt->setFloat(4, item->getPedMeanG6());
    m_writeStmt->setFloat(5, item->getPedMeanG12());
    m_writeStmt->setFloat(6, item->getPedRMSG1());
    m_writeStmt->setFloat(7, item->getPedRMSG6());
    m_writeStmt->setFloat(8, item->getPedRMSG12());
    m_writeStmt->setInt(9, item->getTaskStatus());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalsDat::writeDB():  "+e.getMessage()));
  }
}



void MonPedestalsDat::fetchData(map< EcalLogicID, MonPedestalsDat >* fillMap, MonRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("MonPedestalsDat::writeDB:  IOV not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.ped_mean_g1, d.ped_mean_g6, d.ped_mean_g12, "
		 "d.ped_rms_g1, d.ped_rms_g6, d.ped_rms_g12, d.task_status "
		 "FROM channelview cv JOIN mon_pedestals_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, MonPedestalsDat > p;
    MonPedestalsDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPedMeanG1( rset->getFloat(7) );  
      dat.setPedMeanG6( rset->getFloat(8) );
      dat.setPedMeanG12( rset->getFloat(9) );
      dat.setPedRMSG1( rset->getFloat(10) );
      dat.setPedRMSG6( rset->getFloat(11) );
      dat.setPedRMSG12( rset->getFloat(12) );
      dat.setTaskStatus( rset->getInt(13) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalsDat::fetchData:  "+e.getMessage()));
  }
}

void MonPedestalsDat::writeArrayDB(const std::map< EcalLogicID, MonPedestalsDat >* data, MonRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonPedestalsDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
  float* ww= new float[nrows];
  float* uu= new float[nrows];
  float* tt= new float[nrows];
  int* st= new int[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* w_len= new ub2[nrows];
  ub2* u_len= new ub2[nrows];
  ub2* t_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];

  const EcalLogicID* channel;
  const MonPedestalsDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, MonPedestalsDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("MonPedestalsDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	float x=dataitem->getPedMeanG1();
	float y=dataitem->getPedMeanG6();
	float z=dataitem->getPedMeanG12();
	float w=dataitem->getPedRMSG1();
	float u=dataitem->getPedRMSG6();
	float t=dataitem->getPedRMSG12();
	int statu=dataitem->getTaskStatus();



	xx[count]=x;
	yy[count]=y;
	zz[count]=z;
	ww[count]=w;
	uu[count]=u;
	tt[count]=t;
	st[count]=statu;


	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	w_len[count]=sizeof(ww[count]);
	u_len[count]=sizeof(uu[count]);
	t_len[count]=sizeof(tt[count]);
	st_len[count]=sizeof(st[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)ww, OCCIFLOAT , sizeof(ww[0]), w_len );
    m_writeStmt->setDataBuffer(7, (dvoid*)uu, OCCIFLOAT , sizeof(uu[0]), u_len );
    m_writeStmt->setDataBuffer(8, (dvoid*)tt, OCCIFLOAT , sizeof(tt[0]), t_len );
    m_writeStmt->setDataBuffer(9, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] ww;
    delete [] uu;
    delete [] tt;
    delete [] st;

    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] w_len;
    delete [] u_len;
    delete [] t_len;
    delete [] st_len;



  } catch (SQLException &e) {
    throw(std::runtime_error("MonPedestalsDat::writeArrayDB():  "+e.getMessage()));
  }
}

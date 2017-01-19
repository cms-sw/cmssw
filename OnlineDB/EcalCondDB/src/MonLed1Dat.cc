#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonLed1Dat.h"

using namespace std;
using namespace oracle::occi;

MonLed1Dat::MonLed1Dat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_vptMean = 0;
  m_vptRMS = 0;
  m_vptOverPNMean = 0;
  m_vptOverPNRMS = 0;
  m_taskStatus = 0;
  
}



MonLed1Dat::~MonLed1Dat()
{
}



void MonLed1Dat::prepareWrite()
  noexcept(false)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_led1_dat (iov_id, logic_id, "
			"vpt_mean, vpt_rms, vpt_over_pn_mean, vpt_over_pn_rms, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":vpt_mean, :vpt_rms, :vpt_over_pn_mean, :vpt_over_pn_rms, :task_status)");
  } catch (SQLException &e) {
    throw(std::runtime_error("MonLed1Dat::prepareWrite():  "+e.getMessage()));
  }
}



void MonLed1Dat::writeDB(const EcalLogicID* ecid, const MonLed1Dat* item, MonRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonLed1Dat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("MonLed1Dat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getVPTMean() );
    m_writeStmt->setFloat(4, item->getVPTRMS() );
    m_writeStmt->setFloat(5, item->getVPTOverPNMean() );
    m_writeStmt->setFloat(6, item->getVPTOverPNRMS() );
    m_writeStmt->setInt(7, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("MonLed1Dat::writeDB():  "+e.getMessage()));
  }
}



void MonLed1Dat::fetchData(std::map< EcalLogicID, MonLed1Dat >* fillMap, MonRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();

  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("MonLed1Dat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    
    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.vpt_mean, d.vpt_rms, d.vpt_over_pn_mean, d.vpt_over_pn_rms, d.task_status "
		 "FROM channelview cv JOIN mon_led1_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, MonLed1Dat > p;
    MonLed1Dat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setVPTMean( rset->getFloat(7) );
      dat.setVPTRMS( rset->getFloat(8) );
      dat.setVPTOverPNMean( rset->getFloat(9) );
      dat.setVPTOverPNRMS( rset->getFloat(10) );
      dat.setTaskStatus( rset->getInt(11) );
			

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("MonLed1Dat::fetchData():  "+e.getMessage()));
  }
}

void MonLed1Dat::writeArrayDB(const std::map< EcalLogicID, MonLed1Dat >* data, MonRunIOV* iov)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonLed1Dat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
  float* ww= new float[nrows];
  int* st= new int[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* w_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];

  const EcalLogicID* channel;
  const MonLed1Dat* dataitem;
  int count=0;
  typedef map< EcalLogicID, MonLed1Dat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("MonLed1Dat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	float x=dataitem->getVPTMean();
	float y=dataitem->getVPTRMS();
	float z=dataitem->getVPTOverPNMean();
	float w=dataitem->getVPTOverPNRMS();
	int statu=dataitem->getTaskStatus();



	xx[count]=x;
	yy[count]=y;
	zz[count]=z;
	ww[count]=w;
	st[count]=statu;


	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	w_len[count]=sizeof(ww[count]);
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
    m_writeStmt->setDataBuffer(7, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] ww;
    delete [] st;

    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] w_len;
    delete [] st_len;



  } catch (SQLException &e) {
    throw(std::runtime_error("MonLed1Dat::writeArrayDB():  "+e.getMessage()));
  }
}

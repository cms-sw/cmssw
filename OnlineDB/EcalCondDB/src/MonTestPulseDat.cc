#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/MonTestPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"

using namespace std;
using namespace oracle::occi;

MonTestPulseDat::MonTestPulseDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_adcMeanG1 = 0;
  m_adcRMSG1 = 0;
  m_adcMeanG6 = 0;
  m_adcRMSG6 = 0;
  m_adcMeanG12 = 0;
  m_adcRMSG12 = 0;
  m_taskStatus = 0;
}



MonTestPulseDat::~MonTestPulseDat()
{
}



void MonTestPulseDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO mon_test_pulse_dat (iov_id, logic_id, "
			"adc_mean_g1, adc_rms_g1, adc_mean_g6, adc_rms_g6, adc_mean_g12, adc_rms_g12, task_status) "
			"VALUES (:iov_id, :logic_id, "
			":adc_mean_g1, :adc_rms_g1, :adc_rms_g6, :adc_rms_g6, :adc_mean_g12, :adc_rms_g12, :task_status)");
  } catch (SQLException &e) {
    throw(std::runtime_error("MonTestPulseDat::prepareWrite():  "+e.getMessage()));
  }
}



void MonTestPulseDat::writeDB(const EcalLogicID* ecid, const MonTestPulseDat* item, MonRunIOV* iov )
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonTestPulseDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("MonTestPulseDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getADCMeanG1() );
    m_writeStmt->setFloat(4, item->getADCRMSG1() );
    m_writeStmt->setFloat(5, item->getADCMeanG6() );
    m_writeStmt->setFloat(6, item->getADCRMSG6() );
    m_writeStmt->setFloat(7, item->getADCMeanG12() );
    m_writeStmt->setFloat(8, item->getADCRMSG12() );
    m_writeStmt->setInt(9, item->getTaskStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("MonTestPulseDat::writeDB():  "+e.getMessage()));
  }
}



void MonTestPulseDat::fetchData(std::map< EcalLogicID, MonTestPulseDat >* fillMap, MonRunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("MonTestPulseDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.adc_mean_g1, d.adc_rms_g1, d.adc_mean_g6, d.adc_rms_g6, d.adc_mean_g12, d.adc_rms_g12, d.task_status "
		 "FROM channelview cv JOIN mon_test_pulse_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, MonTestPulseDat > p;
    MonTestPulseDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setADCMeanG1( rset->getFloat(7) );
      dat.setADCRMSG1( rset->getFloat(8) );
      dat.setADCMeanG6( rset->getFloat(9) );
      dat.setADCRMSG6( rset->getFloat(10) );
      dat.setADCMeanG12( rset->getFloat(11) );
      dat.setADCRMSG12( rset->getFloat(12) );
      dat.setTaskStatus( rset->getInt(13) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("MonTestPulseDat::fetchData():  "+e.getMessage()));
  }
}

void MonTestPulseDat::writeArrayDB(const std::map< EcalLogicID, MonTestPulseDat >* data, MonRunIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("MonTestPulseDat::writeArrayDB:  IOV not in DB")); }


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
  const MonTestPulseDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, MonTestPulseDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("MonTestPulseDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	float x=dataitem->getADCMeanG1();
	float y=dataitem->getADCRMSG1();
	float z=dataitem->getADCMeanG6();
	float w=dataitem->getADCRMSG6();
	float u=dataitem->getADCMeanG12();
	float t=dataitem->getADCRMSG12();
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
    throw(std::runtime_error("MonTestPulseDat::writeArrayDB():  "+e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFLaserConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

using namespace std;
using namespace oracle::occi;

LMFLaserConfigDat::LMFLaserConfigDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;


   m_wl=0;
   m_vfe_gain=0;
   m_pn_gain=0;
   m_power=0;
   m_attenuator=0;
   m_current=0;
   m_delay1=0;
   m_delay2=0;
}



LMFLaserConfigDat::~LMFLaserConfigDat()
{
}



void LMFLaserConfigDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_laser_config_dat (lmf_iov_id, logic_id, "
			"wavelength, vfe_gain, pn_gain, lsr_power, lsr_attenuator, lsr_current, lsr_delay_1, lsr_delay_2) "
			"VALUES (:1, :2, "
			":3, :4, :5, :6, :7, :8, :9, :10 )");
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserConfigDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFLaserConfigDat::writeDB(const EcalLogicID* ecid, const LMFLaserConfigDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserConfigDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFLaserConfigDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getWavelength() );
    m_writeStmt->setInt(4, item->getVFEGain() );
    m_writeStmt->setInt(5, item->getPNGain() );
    m_writeStmt->setFloat(6, item->getPower() );
    m_writeStmt->setFloat(7, item->getAttenuator() );
    m_writeStmt->setFloat(8, item->getCurrent() );
    m_writeStmt->setFloat(9, item->getDelay1() );
    m_writeStmt->setFloat(10, item->getDelay2() );
  
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserConfigDat::writeDB():  "+e.getMessage()));
  }
}


void LMFLaserConfigDat::writeArrayDB(const std::map< EcalLogicID, LMFLaserConfigDat >* data, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserConfigDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  
  int* xx= new int[nrows];
  int* yy= new int[nrows];
  int* zz= new int[nrows];
  float* wwa= new float[nrows];
  float* uua= new float[nrows];
  float* tta= new float[nrows];
  float* wwb= new float[nrows];
  float* uub= new float[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* wa_len= new ub2[nrows];
  ub2* ua_len= new ub2[nrows];
  ub2* ta_len= new ub2[nrows];
  ub2* wb_len= new ub2[nrows];
  ub2* ub_len= new ub2[nrows];

  const EcalLogicID* channel;
  const LMFLaserConfigDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, LMFLaserConfigDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(runtime_error("LMFLaserConfigDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);

	int x=dataitem->getWavelength();
	int y=dataitem->getVFEGain();
	int z=dataitem->getPNGain();
	float wa=dataitem->getPower();
	float ua=dataitem->getAttenuator();
	float ta=dataitem->getCurrent();
	float wb=dataitem->getDelay1();
	float ub=dataitem->getDelay2();


	xx[count]=x;
	yy[count]=y;
	zz[count]=z;
	wwa[count]=wa;
	uua[count]=ua;
	tta[count]=ta;
	wwb[count]=wb;
	uub[count]=ub;

	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	wa_len[count]=sizeof(wwa[count]);
	ua_len[count]=sizeof(uua[count]);
	ta_len[count]=sizeof(tta[count]);
	wb_len[count]=sizeof(wwb[count]);
	ub_len[count]=sizeof(uub[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1,  (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2,  (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3,  (dvoid*)xx,  OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4,  (dvoid*)yy,  OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(5,  (dvoid*)zz,  OCCIINT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(6,  (dvoid*)wwa, OCCIFLOAT , sizeof(wwa[0]), wa_len );
    m_writeStmt->setDataBuffer(7,  (dvoid*)uua, OCCIFLOAT , sizeof(uua[0]), ua_len );
    m_writeStmt->setDataBuffer(8,  (dvoid*)tta, OCCIFLOAT , sizeof(tta[0]), ta_len );
    m_writeStmt->setDataBuffer(9, (dvoid*)wwb, OCCIFLOAT , sizeof(wwb[0]), wb_len );
    m_writeStmt->setDataBuffer(10, (dvoid*)uub, OCCIFLOAT , sizeof(uub[0]), ub_len );
   
    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
   
    delete [] xx;
    delete [] yy;
    delete [] zz;
 
    delete [] wwa;
    delete [] uua;
    delete [] tta;
    delete [] wwb;
    delete [] uub;


    delete [] ids_len;
    delete [] iov_len;

    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
 
    delete [] wa_len;
    delete [] ua_len;
    delete [] ta_len;
    delete [] wb_len;
    delete [] ub_len;

  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserConfigDat::writeArrayDB():  "+e.getMessage()));
  }
}

void LMFLaserConfigDat::fetchData(std::map< EcalLogicID, LMFLaserConfigDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFLaserConfigDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
  
    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		       "d.wavelength, d.vfe_gain, d.pn_gain, d.lsr_power, d.lsr_attenuator, d.lsr_current, d.lsr_delay_1, d.lsr_delay_2 "
		       "FROM channelview cv JOIN lmf_laser_config_dat d "
		       "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		       "WHERE d.lmf_iov_id = :iov_id");
 
    m_readStmt->setInt(1, iovID);
    
    ResultSet* rset = m_readStmt->executeQuery();
     
    std::pair< EcalLogicID, LMFLaserConfigDat > p;
    LMFLaserConfigDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to


      dat.setWavelength( rset->getInt(7) );
      dat.setVFEGain( rset->getInt(7) );
      dat.setPNGain( rset->getInt(7) );
      dat.setPower( rset->getFloat(8) );
      dat.setAttenuator( rset->getFloat(8) );
      dat.setCurrent( rset->getFloat(8) );
      dat.setDelay1( rset->getFloat(8) );
      dat.setDelay2( rset->getFloat(8) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserConfigDat::fetchData():  "+e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFPNBluePrimDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

using namespace std;
using namespace oracle::occi;

LMFPNBluePrimDat::LMFPNBluePrimDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;


   m_Mean=0;
   m_RMS=0;
   m_Peak=0;
   m_Flag=0;
   m_PNAOverPNBMean=0;
   m_PNAOverPNBRMS=0;
   m_PNAOverPNBPeak=0;

}



LMFPNBluePrimDat::~LMFPNBluePrimDat()
{
}



void LMFPNBluePrimDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_laser_blue_PN_prim_dat (lmf_iov_id, logic_id, "
			"flag, mean, rms, peak,  pna_over_pnB_mean, pna_over_pnB_rms, pna_over_pnB_peak ) "
			"VALUES (:1, :2, "
			":3, :4, :5, :6, :7, :8, :9 )");
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBluePrimDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFPNBluePrimDat::writeDB(const EcalLogicID* ecid, const LMFPNBluePrimDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFPNBluePrimDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFPNBluePrimDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getFlag() );
    m_writeStmt->setFloat(4, item->getMean() );
    m_writeStmt->setFloat(5, item->getRMS() );
    m_writeStmt->setFloat(6, item->getPeak() );
    m_writeStmt->setFloat(7, item->getPNAOverPNBMean() );
    m_writeStmt->setFloat(8, item->getPNAOverPNBRMS() );
    m_writeStmt->setFloat(9, item->getPNAOverPNBPeak() );
  
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBluePrimDat::writeDB():  "+e.getMessage()));
  }
}


void LMFPNBluePrimDat::writeArrayDB(const std::map< EcalLogicID, LMFPNBluePrimDat >* data, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFPNBluePrimDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  int* aa= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
  float* wwa= new float[nrows];
  float* uua= new float[nrows];
  float* tta= new float[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* a_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* wa_len= new ub2[nrows];
  ub2* ua_len= new ub2[nrows];
  ub2* ta_len= new ub2[nrows];

  const EcalLogicID* channel;
  const LMFPNBluePrimDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, LMFPNBluePrimDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(runtime_error("LMFPNBluePrimDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	int a=dataitem->getFlag();
	float x=dataitem->getMean();
	float y=dataitem->getRMS();
	float z=dataitem->getPeak();
	float wa=dataitem->getPNAOverPNBMean();
	float ua=dataitem->getPNAOverPNBRMS();
	float ta=dataitem->getPNAOverPNBPeak();

	aa[count]=a;
	xx[count]=x;
	yy[count]=y;
	zz[count]=z;
	wwa[count]=wa;
	uua[count]=ua;
	tta[count]=ta;

	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	a_len[count]=sizeof(aa[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	wa_len[count]=sizeof(wwa[count]);
	ua_len[count]=sizeof(uua[count]);
	ta_len[count]=sizeof(tta[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1,  (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2,  (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3,  (dvoid*)aa,  OCCIINT , sizeof(aa[0]), a_len );
    m_writeStmt->setDataBuffer(4,  (dvoid*)xx,  OCCIFLOAT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(5,  (dvoid*)yy,  OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(6,  (dvoid*)zz,  OCCIFLOAT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(7,  (dvoid*)wwa, OCCIFLOAT , sizeof(wwa[0]), wa_len );
    m_writeStmt->setDataBuffer(8,  (dvoid*)uua, OCCIFLOAT , sizeof(uua[0]), ua_len );
    m_writeStmt->setDataBuffer(9,  (dvoid*)tta, OCCIFLOAT , sizeof(tta[0]), ta_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] aa;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] wwa;
    delete [] uua;
    delete [] tta;

    delete [] ids_len;
    delete [] iov_len;
    delete [] a_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] wa_len;
    delete [] ua_len;
    delete [] ta_len;

  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBluePrimDat::writeArrayDB():  "+e.getMessage()));
  }
}

void LMFPNBluePrimDat::fetchData(std::map< EcalLogicID, LMFPNBluePrimDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFPNBluePrimDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
  
    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.flag, d.mean, d.rms, d.peak, d.pna_over_pnb_mean, d.pna_over_pnb_rms, d.pna_over_pnB_peak "
		 "FROM channelview cv JOIN lmf_laser_blue_PN_PRIM_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.lmf_iov_id = :iov_id");
 
    m_readStmt->setInt(1, iovID);
    
    ResultSet* rset = m_readStmt->executeQuery();
     
    std::pair< EcalLogicID, LMFPNBluePrimDat > p;
    LMFPNBluePrimDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setFlag( rset->getInt(7) );
      dat.setMean( rset->getFloat(8) );
      dat.setRMS( rset->getFloat(9) );
      dat.setPeak( rset->getFloat(10) );
      dat.setPNAOverPNBMean( rset->getFloat(11) );
      dat.setPNAOverPNBRMS( rset->getFloat(12) );
      dat.setPNAOverPNBPeak( rset->getFloat(13) );

      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFPNBluePrimDat::fetchData():  "+e.getMessage()));
  }
}

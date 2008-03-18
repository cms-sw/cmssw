#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFLaserBluePrimDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

using namespace std;
using namespace oracle::occi;

LMFLaserBluePrimDat::LMFLaserBluePrimDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;


   m_Mean=0;
   m_RMS=0;
   m_Peak=0;
   m_Flag=0;
   m_apdOverPNAMean=0;
   m_apdOverPNARMS=0;
   m_apdOverPNAPeak=0;
   m_apdOverPNBMean=0;
   m_apdOverPNBRMS=0;
   m_apdOverPNBPeak=0;
   m_apdOverPNMean=0;
   m_apdOverPNRMS=0;
   m_apdOverPNPeak=0;
   m_Alpha=0;
   m_Beta=0;
   m_ShapeCor=0;
}



LMFLaserBluePrimDat::~LMFLaserBluePrimDat()
{
}



void LMFLaserBluePrimDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_laser_blue_prim_dat (lmf_iov_id, logic_id, "
			"flag, mean, rms, peak,  apd_over_pnA_mean, apd_over_pnA_rms, apd_over_pnA_peak, "
			"apd_over_pnB_mean, apd_over_pnB_rms, apd_over_pnB_peak, apd_over_pn_mean, apd_over_pn_rms, apd_over_pn_peak, "
                        " alpha, beta, shape_cor ) "
			"VALUES (:1, :2, "
			":3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18 )");
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserBluePrimDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFLaserBluePrimDat::writeDB(const EcalLogicID* ecid, const LMFLaserBluePrimDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserBluePrimDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFLaserBluePrimDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getFlag() );
    m_writeStmt->setFloat(4, item->getMean() );
    m_writeStmt->setFloat(5, item->getRMS() );
    m_writeStmt->setFloat(6, item->getPeak() );
    m_writeStmt->setFloat(7, item->getAPDOverPNAMean() );
    m_writeStmt->setFloat(8, item->getAPDOverPNARMS() );
    m_writeStmt->setFloat(9, item->getAPDOverPNAPeak() );
    m_writeStmt->setFloat(10, item->getAPDOverPNBMean() );
    m_writeStmt->setFloat(11, item->getAPDOverPNBRMS() );
    m_writeStmt->setFloat(12, item->getAPDOverPNBPeak() );
    m_writeStmt->setFloat(13, item->getAPDOverPNMean() );
    m_writeStmt->setFloat(14, item->getAPDOverPNRMS() );
    m_writeStmt->setFloat(15, item->getAPDOverPNPeak() );
    m_writeStmt->setFloat(16, item->getAlpha() );
    m_writeStmt->setFloat(17, item->getBeta() );
    m_writeStmt->setFloat(18, item->getShapeCor() );
  
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserBluePrimDat::writeDB():  "+e.getMessage()));
  }
}


void LMFLaserBluePrimDat::writeArrayDB(const std::map< EcalLogicID, LMFLaserBluePrimDat >* data, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFLaserBluePrimDat::writeArrayDB:  IOV not in DB")); }


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
  float* wwb= new float[nrows];
  float* uub= new float[nrows];
  float* ttb= new float[nrows];
  float* ww= new float[nrows];
  float* uu= new float[nrows];
  float* tt= new float[nrows];
  float* ualpha= new float[nrows];
  float* ubeta= new float[nrows];
  float* ushapecor= new float[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* a_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* wa_len= new ub2[nrows];
  ub2* ua_len= new ub2[nrows];
  ub2* ta_len= new ub2[nrows];
  ub2* wb_len= new ub2[nrows];
  ub2* ub_len= new ub2[nrows];
  ub2* tb_len= new ub2[nrows];
  ub2* w_len= new ub2[nrows];
  ub2* u_len= new ub2[nrows];
  ub2* t_len= new ub2[nrows];
  ub2* ualpha_len= new ub2[nrows];
  ub2* ubeta_len= new ub2[nrows];
  ub2* ushapecor_len= new ub2[nrows];

  const EcalLogicID* channel;
  const LMFLaserBluePrimDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, LMFLaserBluePrimDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(runtime_error("LMFLaserBluePrimDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	int a=dataitem->getFlag();
	float x=dataitem->getMean();
	float y=dataitem->getRMS();
	float z=dataitem->getPeak();
	float wa=dataitem->getAPDOverPNAMean();
	float ua=dataitem->getAPDOverPNARMS();
	float ta=dataitem->getAPDOverPNAPeak();
	float wb=dataitem->getAPDOverPNBMean();
	float ub=dataitem->getAPDOverPNBRMS();
	float tb=dataitem->getAPDOverPNBPeak();
	float w=dataitem->getAPDOverPNMean();
	float u=dataitem->getAPDOverPNRMS();
	float t=dataitem->getAPDOverPNPeak();
	float alpha=dataitem->getAlpha();
	float beta=dataitem->getBeta();
	float shapecor=dataitem->getShapeCor();


	aa[count]=a;
	xx[count]=x;
	yy[count]=y;
	zz[count]=z;
	wwa[count]=wa;
	uua[count]=ua;
	tta[count]=ta;
	wwb[count]=wb;
	uub[count]=ub;
	ttb[count]=tb;
	ww[count]=w;
	uu[count]=u;
	tt[count]=t;
	ualpha[count]=alpha;
	ubeta[count]=beta;
	ushapecor[count]=shapecor;

	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	a_len[count]=sizeof(aa[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	wa_len[count]=sizeof(wwa[count]);
	ua_len[count]=sizeof(uua[count]);
	ta_len[count]=sizeof(tta[count]);
	wb_len[count]=sizeof(wwb[count]);
	ub_len[count]=sizeof(uub[count]);
	tb_len[count]=sizeof(ttb[count]);
	w_len[count]=sizeof(ww[count]);
	u_len[count]=sizeof(uu[count]);
	t_len[count]=sizeof(tt[count]);
	ualpha_len[count]=sizeof(ualpha[count]);
	ubeta_len[count]=sizeof(ubeta[count]);
	ushapecor_len[count]=sizeof(ushapecor[count]);

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
    m_writeStmt->setDataBuffer(10, (dvoid*)wwb, OCCIFLOAT , sizeof(wwb[0]), wb_len );
    m_writeStmt->setDataBuffer(11, (dvoid*)uub, OCCIFLOAT , sizeof(uub[0]), ub_len );
    m_writeStmt->setDataBuffer(12, (dvoid*)ttb, OCCIFLOAT , sizeof(ttb[0]), tb_len );
    m_writeStmt->setDataBuffer(13, (dvoid*)ww,  OCCIFLOAT , sizeof(ww[0]),   w_len );
    m_writeStmt->setDataBuffer(14, (dvoid*)uu,  OCCIFLOAT , sizeof(uu[0]),   u_len );
    m_writeStmt->setDataBuffer(15, (dvoid*)tt,  OCCIFLOAT , sizeof(tt[0]),   t_len );
    m_writeStmt->setDataBuffer(16, (dvoid*)ualpha, OCCIFLOAT , sizeof(ualpha[0]), ualpha_len );
    m_writeStmt->setDataBuffer(17, (dvoid*)ubeta,  OCCIFLOAT , sizeof(ubeta[0]),  ubeta_len );
    m_writeStmt->setDataBuffer(18, (dvoid*)ushapecor,  OCCIFLOAT , sizeof(ushapecor[0]),  ushapecor_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] aa;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] ww;
    delete [] uu;
    delete [] tt;
    delete [] wwa;
    delete [] uua;
    delete [] tta;
    delete [] wwb;
    delete [] uub;
    delete [] ttb;
    delete [] ualpha;
    delete [] ubeta;
    delete [] ushapecor;

    delete [] ids_len;
    delete [] iov_len;
    delete [] a_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] w_len;
    delete [] u_len;
    delete [] t_len;
    delete [] wa_len;
    delete [] ua_len;
    delete [] ta_len;
    delete [] wb_len;
    delete [] ub_len;
    delete [] tb_len;
    delete [] ualpha_len;
    delete [] ubeta_len;
    delete [] ushapecor_len;
   



  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserBluePrimDat::writeArrayDB():  "+e.getMessage()));
  }
}

void LMFLaserBluePrimDat::fetchData(std::map< EcalLogicID, LMFLaserBluePrimDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFLaserBluePrimDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
  
    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.flag, d.mean, d.rms, d.peak, d.apd_over_pnA_mean, d.apd_over_pnA_rms, d.apd_over_pnA_peak, "
         " d.apd_over_pnB_mean, d.apd_over_pnB_rms,d.apd_over_pnB_peak, d.apd_over_pn_mean, d.apd_over_pn_rms, d.apd_over_pn_peak "
		 "d.alpha, d.beta, d.shape_cor "
		 "FROM channelview cv JOIN lmf_laser_blue_prim_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.lmf_iov_id = :iov_id");
 
    m_readStmt->setInt(1, iovID);
    
    ResultSet* rset = m_readStmt->executeQuery();
     
    std::pair< EcalLogicID, LMFLaserBluePrimDat > p;
    LMFLaserBluePrimDat dat;
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

      dat.setAPDOverPNAMean( rset->getFloat(11) );
      dat.setAPDOverPNARMS( rset->getFloat(12) );
      dat.setAPDOverPNAPeak( rset->getFloat(13) );

      dat.setAPDOverPNBMean( rset->getFloat(14) );
      dat.setAPDOverPNBRMS( rset->getFloat(15) );
      dat.setAPDOverPNBPeak( rset->getFloat(16) );

      dat.setAPDOverPNMean( rset->getFloat(17) );
      dat.setAPDOverPNRMS( rset->getFloat(18) );
      dat.setAPDOverPNPeak( rset->getFloat(19) );

      dat.setAlpha( rset->getFloat(20) );
      dat.setBeta( rset->getFloat(21) );
      dat.setShapeCor( rset->getFloat(22) );
      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFLaserBluePrimDat::fetchData():  "+e.getMessage()));
  }
}

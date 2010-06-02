#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/LMFTestPulseConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

using namespace std;
using namespace oracle::occi;

LMFTestPulseConfigDat::LMFTestPulseConfigDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;


   m_vfe_gain=0;
   m_dac_mgpa=0;
   m_pn_gain=0;
   m_pn_vinj=0;

}



LMFTestPulseConfigDat::~LMFTestPulseConfigDat()
{
}



void LMFTestPulseConfigDat::prepareWrite()
  throw(runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO lmf_test_pulse_config_dat (lmf_iov_id, logic_id, "
			"vfe_gain, dac_mgpa, pn_gain, pn_vinj ) "
			"VALUES (:1, :2, "
			":3, :4, :5, :6 )");
  } catch (SQLException &e) {
    throw(runtime_error("LMFTestPulseConfigDat::prepareWrite():  "+e.getMessage()));
  }
}



void LMFTestPulseConfigDat::writeDB(const EcalLogicID* ecid, const LMFTestPulseConfigDat* item, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFTestPulseConfigDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(runtime_error("LMFTestPulseConfigDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setInt(3, item->getVFEGain() );
    m_writeStmt->setInt(3, item->getDACMGPA() );
    m_writeStmt->setInt(3, item->getPNGain() );
    m_writeStmt->setInt(3, item->getPNVinj() );
  
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(runtime_error("LMFTestPulseConfigDat::writeDB():  "+e.getMessage()));
  }
}


void LMFTestPulseConfigDat::writeArrayDB(const std::map< EcalLogicID, LMFTestPulseConfigDat >* data, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(runtime_error("LMFTestPulseConfigDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  int* aa= new int[nrows];
  int* xx= new int[nrows];
  int* yy= new int[nrows];
  int* zz= new int[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* a_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];


  const EcalLogicID* channel;
  const LMFTestPulseConfigDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, LMFTestPulseConfigDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(runtime_error("LMFTestPulseConfigDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	int a=dataitem->getVFEGain();
	int x=dataitem->getDACMGPA();
	int y=dataitem->getPNGain();
	int z=dataitem->getPNVinj();


	aa[count]=a;
	xx[count]=x;
	yy[count]=y;
	zz[count]=z;


	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	a_len[count]=sizeof(aa[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);


	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1,  (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2,  (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3,  (dvoid*)aa,  OCCIINT , sizeof(aa[0]), a_len );
    m_writeStmt->setDataBuffer(4,  (dvoid*)xx,  OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(5,  (dvoid*)yy,  OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(6,  (dvoid*)zz,  OCCIINT , sizeof(zz[0]), z_len );


    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] aa;
    delete [] xx;
    delete [] yy;
    delete [] zz;


    delete [] ids_len;
    delete [] iov_len;
    delete [] a_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;


  } catch (SQLException &e) {
    throw(runtime_error("LMFTestPulseConfigDat::writeArrayDB():  "+e.getMessage()));
  }
}

void LMFTestPulseConfigDat::fetchData(std::map< EcalLogicID, LMFTestPulseConfigDat >* fillMap, LMFRunIOV* iov)
  throw(runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(runtime_error("LMFTestPulseConfigDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
  
    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.vfe_gain, d.dac_mgpa, d.pn_gain, d.pn_vinj, "
		 "FROM channelview cv JOIN lmf_test_pulse_config_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.lmf_iov_id = :iov_id");
 
    m_readStmt->setInt(1, iovID);
    
    ResultSet* rset = m_readStmt->executeQuery();
     
    std::pair< EcalLogicID, LMFTestPulseConfigDat > p;
    LMFTestPulseConfigDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setVFEGain( rset->getInt(7) );
      dat.setDACMGPA( rset->getInt(8) );
      dat.setPNGain( rset->getInt(9) );
      dat.setPNVinj( rset->getInt(10) );


      p.second = dat;
      fillMap->insert(p);
    }

  } catch (SQLException &e) {
    throw(runtime_error("LMFTestPulseConfigDat::fetchData():  "+e.getMessage()));
  }
}

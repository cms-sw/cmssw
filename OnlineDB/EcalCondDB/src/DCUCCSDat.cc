#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCUCCSDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCUCCSDat::DCUCCSDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_m1_vdd1 = 0;
  m_m2_vdd1 = 0;
  m_m1_vdd2 = 0;
  m_m2_vdd2 = 0;
  m_m1_vinj = 0;
  m_m2_vinj = 0;
  m_m1_vcc = 0;
  m_m2_vcc = 0;
  m_m1_dcutemp = 0;
  m_m2_dcutemp = 0;
  m_ccstemplow = 0;
  m_ccstemphigh = 0;
}

DCUCCSDat::~DCUCCSDat()
{
}

void DCUCCSDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO dcu_ccs_dat (iov_id, logic_id, "
			"m1_vdd1, m2_vdd1, m1_vdd2, m2_vdd2, m1_vinj, "
			"m2_vinj, m1_vcc, m2_vcc, m1_dcutemp, m2_dcutemp, "
			"ccstemplow, ccstemphigh) "
			"VALUES (:iov_id, :logic_id, "
			":m1_vdd1, :m2_vdd1, :m1_vdd2, :m2_vdd2, :m1_vinj, "
			":m2_vinj, :m1_vcc, :m2_vcc, :m1_dcutemp, "
			":m2_dcutemp, :ccstemplow, :ccstemphigh)");
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUCCSDat::prepareWrite():  " + 
			     e.getMessage()));
  }
}



void DCUCCSDat::writeDB(const EcalLogicID* ecid, const DCUCCSDat* item, 
			DCUIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { 
    throw(std::runtime_error("DCUCCSDat::writeDB:  IOV not in DB")); 
  }

  int logicID = ecid->getLogicID();
  if (!logicID) { 
    throw(std::runtime_error("DCUCCSDat::writeDB:  Bad EcalLogicID")); 
  }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat( 3, item->getM1VDD1() );
    m_writeStmt->setFloat( 4, item->getM2VDD1() );
    m_writeStmt->setFloat( 5, item->getM1VDD2() );
    m_writeStmt->setFloat( 6, item->getM2VDD2() );
    m_writeStmt->setFloat( 7, item->getM1Vinj() );
    m_writeStmt->setFloat( 8, item->getM2Vinj() );
    m_writeStmt->setFloat( 9, item->getM1Vcc() );
    m_writeStmt->setFloat(10, item->getM2Vcc() );
    m_writeStmt->setFloat(11, item->getM1DCUTemp() );
    m_writeStmt->setFloat(12, item->getM2DCUTemp() );
    m_writeStmt->setFloat(13, item->getCCSTempLow() );
    m_writeStmt->setFloat(14, item->getCCSTempHigh() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUCCSDat::writeDB():  " + e.getMessage()));
  }
}

void DCUCCSDat::fetchData(std::map< EcalLogicID, DCUCCSDat >* fillMap, 
			  DCUIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("DCUCCSDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, "
		       "cv.maps_to, "
		       "d.m1_vdd1, d.m2_vdd1, d.m1_vdd2, d.m2_vdd2, "
		       "d.m1_vinj, d.m2_vinj, "
		       "d.m1_vcc, d.m2_vcc, "
		       "d.m1_dcutemp, d.m2_dcutemp, "
		       "d.ccstemplow, d.ccstemphigh "
		       "FROM channelview cv JOIN dcu_ccs_dat d "
		       "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		       "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, DCUCCSDat > p;
    DCUCCSDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setVDD(rset->getFloat(7), rset->getFloat(9), rset->getFloat(8),
		  rset->getFloat(10));
      dat.setVinj(rset->getFloat(11), rset->getFloat(12));
      dat.setVcc(rset->getFloat(13), rset->getFloat(14));
      dat.setDCUTemp(rset->getFloat(15), rset->getFloat(16));
      dat.setCCSTemp(rset->getFloat(17), rset->getFloat(18));
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("DCUCCSDat::fetchData():  "+e.getMessage()));
  }
}

void DCUCCSDat::writeArrayDB(const std::map< EcalLogicID, DCUCCSDat >* data, DCUIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { 
    throw(std::runtime_error("DCUCCSDat::writeArrayDB:  IOV not in DB")); 
  }

  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  float* x1= new float[nrows];
  float* x2= new float[nrows];
  float* x3= new float[nrows];
  float* x4= new float[nrows];
  float* x5= new float[nrows];
  float* x6= new float[nrows];
  float* x7= new float[nrows];
  float* x8= new float[nrows];
  float* x9= new float[nrows];
  float* xa= new float[nrows];
  float* xb= new float[nrows];
  float* xc= new float[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len  = new ub2[nrows];

  const EcalLogicID* channel;
  const DCUCCSDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, DCUCCSDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { 
	  throw(std::runtime_error("DCUCCSDat::writeArrayDB:  Bad EcalLogicID")); 
	}
	ids[count]=logicID;
	iovid_vec[count]=iovID;
	
	dataitem = &(p->second);

	x1[count] = dataitem->getM1VDD1();
	x2[count] = dataitem->getM2VDD1();
	x3[count] = dataitem->getM1VDD2();
	x4[count] = dataitem->getM2VDD2();
	x5[count] = dataitem->getM1Vinj();
	x6[count] = dataitem->getM2Vinj();
	x7[count] = dataitem->getM1Vcc();
	x8[count] = dataitem->getM2Vcc();
	x9[count] = dataitem->getM1DCUTemp();
	xa[count] = dataitem->getM2DCUTemp();
	xb[count] = dataitem->getCCSTempLow();
	xc[count] = dataitem->getCCSTempHigh();

	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(x1[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, 
			       sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, 
			       sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x1, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x2, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x3, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x4, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x5, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x6, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x7, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x8, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)x9, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xa, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xb, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xc, OCCIFLOAT , 
			       sizeof(x1[0]), x_len );
   
    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] x1;
    delete [] x2;
    delete [] x3;
    delete [] x4;
    delete [] x5;
    delete [] x6;
    delete [] x7;
    delete [] x8;
    delete [] x9;
    delete [] xa;
    delete [] xb;
    delete [] xc;

    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;


  } catch (SQLException &e) {
    throw(std::runtime_error("DCUCCSDat::writeArrayDB():  " + e.getMessage()));
  }
}

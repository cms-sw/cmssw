#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCULVRBTempsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

using namespace std;
using namespace oracle::occi;

DCULVRBTempsDat::DCULVRBTempsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_t1 = 0;
  m_t2 = 0;
  m_t3 = 0;
}



DCULVRBTempsDat::~DCULVRBTempsDat()
{
}



void DCULVRBTempsDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO dcu_lvrb_temps_dat (iov_id, logic_id, "
			"t1, t2, t3) "
			"VALUES (:iov_id, :logic_id, "
			":3, :4, :5)");
  } catch (SQLException &e) {
    throw(std::runtime_error("DCULVRBTempsDat::prepareWrite():  "+e.getMessage()));
  }
}



void DCULVRBTempsDat::writeDB(const EcalLogicID* ecid, const DCULVRBTempsDat* item, DCUIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("DCULVRBTempsDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("DCULVRBTempsDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getT1() );
    m_writeStmt->setFloat(4, item->getT2() );
    m_writeStmt->setFloat(5, item->getT3() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("DCULVRBTempsDat::writeDB():  "+e.getMessage()));
  }
}



void DCULVRBTempsDat::fetchData(std::map< EcalLogicID, DCULVRBTempsDat >* fillMap, DCUIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("DCULVRBTempsDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.t1, d.t2, d.t3 "
		 "FROM channelview cv JOIN dcu_lvrb_temps_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, DCULVRBTempsDat > p;
    DCULVRBTempsDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setT1( rset->getFloat(7) );
      dat.setT2( rset->getFloat(8) );
      dat.setT3( rset->getFloat(9) );

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("DCULVRBTempsDat::fetchData():  "+e.getMessage()));
  }
}

void DCULVRBTempsDat::writeArrayDB(const std::map< EcalLogicID, DCULVRBTempsDat >* data, DCUIOV* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("DCULVRBTempsDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
 
  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];


  const EcalLogicID* channel;
  const DCULVRBTempsDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, DCULVRBTempsDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("DCULVRBTempsDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iov);
	float x=dataitem->getT1();
	float y=dataitem->getT2();
	float z=dataitem->getT3();



	xx[count]=x;
	yy[count]=y;
	zz[count]=z;


	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);


	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT , sizeof(zz[0]), z_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;

    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;


  } catch (SQLException &e) {
    throw(std::runtime_error("DCULVRBTempsDat::writeArrayDB():  "+e.getMessage()));
  }
}

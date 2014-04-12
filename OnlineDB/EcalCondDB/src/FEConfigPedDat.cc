#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigPedDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigPedInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigPedDat::FEConfigPedDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;
  m_ID=0;
  m_pedMeanG1 = 0;
  m_pedMeanG6 = 0;
  m_pedMeanG12 = 0;

}



FEConfigPedDat::~FEConfigPedDat()
{
}


void FEConfigPedDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();

    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (ped_conf_id, logic_id, "
		      "mean_12, mean_6, mean_1 ) "
		      "VALUES (:ped_conf_id, :logic_id, "
		      ":ped_mean_g12, :ped_mean_g6, :ped_mean_g1 )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigPedDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigPedDat::writeDB(const EcalLogicID* ecid, const FEConfigPedDat* item, FEConfigPedInfo* iconf )
  throw(std::runtime_error)
{
  this->checkConnection();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigPedDat::writeDB:  ICONF not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("FEConfigPedDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getPedMeanG12());
    m_writeStmt->setFloat(4, item->getPedMeanG6());
    m_writeStmt->setFloat(5, item->getPedMeanG1());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigPedDat::writeDB():  "+e.getMessage()));
  }
}


void FEConfigPedDat::fetchData(map< EcalLogicID, FEConfigPedDat >* fillMap, FEConfigPedInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
    //  throw(std::runtime_error("FEConfigPedDat::writeDB:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.mean_12, d.mean_6, d.mean_1 "
		 "FROM channelview cv JOIN fe_config_ped_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE ped_conf_id = :ped_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigPedDat > p;
    FEConfigPedDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setPedMeanG12( rset->getFloat(7) );  
      dat.setPedMeanG6( rset->getFloat(8) );
      dat.setPedMeanG1( rset->getFloat(9) );
    
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigPedDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigPedDat::writeArrayDB(const std::map< EcalLogicID, FEConfigPedDat >* data, FEConfigPedInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigPedDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iconfid_vec= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];


  const EcalLogicID* channel;
  const FEConfigPedDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigPedDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("FEConfigPedDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iconfid_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iconf);
	float x=dataitem->getPedMeanG12();
	float y=dataitem->getPedMeanG6();
	float z=dataitem->getPedMeanG1();

	xx[count]=x;
	yy[count]=y;
	zz[count]=z;

	ids_len[count]=sizeof(ids[count]);
	iconf_len[count]=sizeof(iconfid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)zz, OCCIFLOAT , sizeof(zz[0]), z_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iconfid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;

    delete [] ids_len;
    delete [] iconf_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigPedDat::writeArrayDB():  "+e.getMessage()));
  }
}

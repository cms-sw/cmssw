#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSlidingInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigSlidingDat::FEConfigSlidingDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_sliding = 0;

}



FEConfigSlidingDat::~FEConfigSlidingDat()
{
}



void FEConfigSlidingDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO fe_config_sliding_dat (sli_conf_id, logic_id, "
		      "sliding ) "
		      "VALUES (:sli_conf_id, :logic_id, "
		      ":sliding )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigSlidingDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigSlidingDat::writeDB(const EcalLogicID* ecid, const FEConfigSlidingDat* item, FEConfigSlidingInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigSlidingDat::writeDB:  ICONF not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("FEConfigSlidingDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getSliding());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigSlidingDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigSlidingDat::fetchData(map< EcalLogicID, FEConfigSlidingDat >* fillMap, FEConfigSlidingInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
    //  throw(std::runtime_error("FEConfigSlidingDat::writeDB:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.sliding "
		 "FROM channelview cv JOIN fe_config_sliding_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE sli_conf_id = :sli_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigSlidingDat > p;
    FEConfigSlidingDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setSliding( rset->getFloat(7) );  

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigSlidingDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigSlidingDat::writeArrayDB(const std::map< EcalLogicID, FEConfigSlidingDat >* data, FEConfigSlidingInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigSlidingDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iconfid_vec= new int[nrows];
  float* xx= new float[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];


  const EcalLogicID* channel;
  const FEConfigSlidingDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigSlidingDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("FEConfigSlidingDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iconfid_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iconf);
	float x=dataitem->getSliding();

	xx[count]=x;


	ids_len[count]=sizeof(ids[count]);
	iconf_len[count]=sizeof(iconfid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);


	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT , sizeof(xx[0]), x_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iconfid_vec;
    delete [] xx;

    delete [] ids_len;
    delete [] iconf_len;
    delete [] x_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigSlidingDat::writeArrayDB():  "+e.getMessage()));
  }
}

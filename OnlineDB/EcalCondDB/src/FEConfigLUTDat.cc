#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLUTDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigLUTDat::FEConfigLUTDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_group_id = 0;

}



FEConfigLUTDat::~FEConfigLUTDat()
{
}



void FEConfigLUTDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO fe_config_lut_dat (lut_conf_id, logic_id, "
		      "group_id ) "
		      "VALUES (:lut_conf_id, :logic_id, "
		      ":group_id )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTDat::prepareWrite():  "+e.getMessage()));
  }
}


void FEConfigLUTDat::writeDB(const EcalLogicID* ecid, const FEConfigLUTDat* item, FEConfigLUTInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigLUTDat::writeDB:  ICONF not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("FEConfigLUTDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setInt(3, item->getLUTGroupId());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigLUTDat::fetchData(map< EcalLogicID, FEConfigLUTDat >* fillMap, FEConfigLUTInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
    //  throw(std::runtime_error("FEConfigLUTDat::writeDB:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.group_id  "
		 "FROM channelview cv JOIN fe_config_lut_dat d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		       "WHERE lut_conf_id = :lut_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigLUTDat > p;
    FEConfigLUTDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setLUTGroupId( rset->getInt(7) );  
       
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigLUTDat::writeArrayDB(const std::map< EcalLogicID, FEConfigLUTDat >* data, FEConfigLUTInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigLUTDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iconfid_vec= new int[nrows];
  int* xx= new int[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];


  const EcalLogicID* channel;
  const FEConfigLUTDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigLUTDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("FEConfigLUTDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iconfid_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iconf);
	int x=dataitem->getLUTGroupId();

	xx[count]=x;

	ids_len[count]=sizeof(ids[count]);
	iconf_len[count]=sizeof(iconfid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iconfid_vec;
    delete [] xx;

    delete [] ids_len;
    delete [] iconf_len;
    delete [] x_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTDat::writeArrayDB():  "+e.getMessage()));
  }
}

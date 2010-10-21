#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigFgrEEStripDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigFgrEEStripDat::FEConfigFgrEEStripDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_thresh = 0;
  m_lut_fg = 0;

}



FEConfigFgrEEStripDat::~FEConfigFgrEEStripDat()
{
}



void FEConfigFgrEEStripDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (fgr_conf_id, logic_id, "
		      "threshold, lut_fg ) "
		      "VALUES (:fgr_conf_id, :logic_id, "
		      ":threshold, :lut_fg )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrEEStripDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigFgrEEStripDat::writeDB(const EcalLogicID* ecid, const FEConfigFgrEEStripDat* item, FEConfigFgrInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigFgrEEStripDat::writeDB:  ICONF not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("FEConfigFgrEEStripDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setUInt(3, item->getThreshold());
    m_writeStmt->setUInt(4, item->getLutFg());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrEEStripDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigFgrEEStripDat::fetchData(map< EcalLogicID, FEConfigFgrEEStripDat >* fillMap, FEConfigFgrInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
    //  throw(std::runtime_error("FEConfigFgrEEStripDat::writeDB:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.threshold, d.lut_fg "
		 "FROM channelview cv JOIN "+getTable()+" d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE fgr_conf_id = :fgr_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigFgrEEStripDat > p;
    FEConfigFgrEEStripDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setThreshold( rset->getUInt(7) );  
      dat.setLutFg( rset->getUInt(8) );  

      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrEEStripDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigFgrEEStripDat::writeArrayDB(const std::map< EcalLogicID, FEConfigFgrEEStripDat >* data, FEConfigFgrInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigFgrEEStripDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iconfid_vec= new int[nrows];
  unsigned int* xx= new unsigned int[nrows];
  unsigned int* yy= new unsigned int[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];


  const EcalLogicID* channel;
  const FEConfigFgrEEStripDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigFgrEEStripDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("FEConfigFgrEEStripDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iconfid_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iconf);
	unsigned int x=dataitem->getThreshold();
	unsigned int y=dataitem->getLutFg();

	xx[count]=x;
	yy[count]=y;


	ids_len[count]=sizeof(ids[count]);
	iconf_len[count]=sizeof(iconfid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIUNSIGNED_INT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIUNSIGNED_INT , sizeof(yy[0]), y_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iconfid_vec;
    delete [] xx;
    delete [] yy;

    delete [] ids_len;
    delete [] iconf_len;
    delete [] x_len;
    delete [] y_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrEEStripDat::writeArrayDB():  "+e.getMessage()));
  }
}

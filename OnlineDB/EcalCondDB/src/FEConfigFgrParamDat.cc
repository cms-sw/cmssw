#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigFgrParamDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigFgrParamDat::FEConfigFgrParamDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_fglowthresh = 0;
  m_fghighthresh = 0;
  m_lowratio = 0;
  m_highratio = 0;

}



FEConfigFgrParamDat::~FEConfigFgrParamDat()
{
}



void FEConfigFgrParamDat::prepareWrite()
  noexcept(false)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+ getTable()+" (fgr_conf_id, logic_id, "
		      " fg_lowthresh, fg_highthresh, fg_lowratio, fg_highratio ) "
		      "VALUES (:fgr_conf_id, :logic_id, "
		      " :fg_lowthresh, :fg_highthresh, :fg_lowratio, :fg_highratio )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrParamDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigFgrParamDat::writeDB(const EcalLogicID* ecid, const FEConfigFgrParamDat* item, FEConfigFgrInfo* iconf)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigFgrParamDat::writeDB:  ICONF not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("FEConfigFgrParamDat::writeDB:  Bad EcalLogicID")); }
 
  try {
    m_writeStmt->setInt(1, iconfID);
    m_writeStmt->setInt(2, logicID);
    m_writeStmt->setFloat(3, item->getFGlowthresh());
    m_writeStmt->setFloat(4, item->getFGhighthresh());
    m_writeStmt->setFloat(5, item->getFGlowratio());
    m_writeStmt->setFloat(6, item->getFGhighratio());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrParamDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigFgrParamDat::fetchData(map< EcalLogicID, FEConfigFgrParamDat >* fillMap, FEConfigFgrInfo* iconf)
  noexcept(false)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
    //  throw(std::runtime_error("FEConfigFgrParamDat::writeDB:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 " d.fg_lowthresh, d.fg_highthresh, d.fg_lowratio, d.fg_highratio "
		 "FROM channelview cv JOIN "+ getTable() +" d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE fgr_conf_id = :fgr_conf_id");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigFgrParamDat > p;
    FEConfigFgrParamDat dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setFGlowthresh( rset->getFloat(7) );  
      dat.setFGhighthresh(  rset->getFloat(8) );
      dat.setFGlowratio(  rset->getFloat(9) );
      dat.setFGhighratio(  rset->getFloat(10) );
    
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrParamDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigFgrParamDat::writeArrayDB(const std::map< EcalLogicID, FEConfigFgrParamDat >* data, FEConfigFgrInfo* iconf)
  noexcept(false)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigFgrParamDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iov_vec= new int[nrows];
  float* ww= new float[nrows];
  float* uu= new float[nrows];
  float* tt= new float[nrows];
  float* st= new float[nrows];

  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* w_len= new ub2[nrows];
  ub2* u_len= new ub2[nrows];
  ub2* t_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];

  const EcalLogicID* channel;
  const FEConfigFgrParamDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigFgrParamDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("FEConfigFgrParamDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iov_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, conf);
	float w=dataitem->getFGlowthresh();
	float u=dataitem->getFGhighthresh();
	float t=dataitem->getFGlowratio();
	float r=dataitem->getFGhighratio();

	ww[count]=w;
	uu[count]=u;
	tt[count]=t;
	st[count]=r;


	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iov_vec[count]);
	
	w_len[count]=sizeof(ww[count]);
	u_len[count]=sizeof(uu[count]);
	t_len[count]=sizeof(tt[count]);
	st_len[count]=sizeof(st[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iov_vec, OCCIINT, sizeof(iov_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)ww, OCCIFLOAT , sizeof(ww[0]), w_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)uu, OCCIFLOAT , sizeof(uu[0]), u_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)tt, OCCIFLOAT , sizeof(tt[0]), t_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)st, OCCIFLOAT , sizeof(st[0]), st_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iov_vec;
    delete [] ww;
    delete [] uu;
    delete [] tt;
    delete [] st;

    delete [] ids_len;
    delete [] iov_len;
    delete [] w_len;
    delete [] u_len;
    delete [] t_len;
    delete [] st_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrParamDat::writeArrayDB():  "+e.getMessage()));
  }
}

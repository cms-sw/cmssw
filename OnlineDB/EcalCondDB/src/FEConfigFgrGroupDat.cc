#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigFgrGroupDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigFgrInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigFgrGroupDat::FEConfigFgrGroupDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_group_id=0;
  m_thresh_low = 0;
  m_thresh_high = 0;
  m_ratio_low = 0;
  m_ratio_high = 0;
  m_lut = 0;

}



FEConfigFgrGroupDat::~FEConfigFgrGroupDat()
{
}



void FEConfigFgrGroupDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO fe_fgr_per_group_dat (fgr_conf_id, group_id, "
		      " threshold_low, threshold_high, ratio_low, ratio_high, lut_value ) "
		      "VALUES (:fgr_conf_id, :group_id, "
		      ":3, :4, :5, :6, :7 )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrGroupDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigFgrGroupDat::writeDB(const EcalLogicID* ecid, const FEConfigFgrGroupDat* item, FEConfigFgrInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigFgrGroupDat::writeDB:  ICONF not in DB")); }
  /* no need for the logic id in this table 
     int logicID = ecid->getLogicID();
     if (!logicID) { throw(std::runtime_error("FEConfigFgrGroupDat::writeDB:  Bad EcalLogicID")); }
  */

  try {
    m_writeStmt->setInt(1, iconfID);

    m_writeStmt->setInt(2, item->getFgrGroupId());
    m_writeStmt->setFloat(3, item->getThreshLow());
    m_writeStmt->setFloat(4, item->getThreshHigh());
    m_writeStmt->setFloat(5, item->getRatioLow());
    m_writeStmt->setFloat(6, item->getRatioHigh());
    m_writeStmt->setInt(7, item->getLUTValue());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrGroupDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigFgrGroupDat::fetchData(map< EcalLogicID, FEConfigFgrGroupDat >* fillMap, FEConfigFgrInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
     throw(std::runtime_error("FEConfigFgrGroupDat::fetchData:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT d.group_id, d.threshold_low, d.threshold_high, d.ratio_low, d.ratio_high, d.lut_value  "
		 "FROM fe_fgr_per_group_dat d "
		 "WHERE fgr_conf_id = :fgr_conf_id order by d.group_id ");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigFgrGroupDat > p;
    FEConfigFgrGroupDat dat;
    int ig=-1;
    while(rset->next()) {
      ig++;                          // we create a dummy logic_id
      p.first = EcalLogicID( "Group_id",     // name
			     ig );        // logic_id
			   
      dat.setFgrGroupId( rset->getInt(1) );  
      dat.setThreshLow( rset->getFloat(2) );  
      dat.setThreshHigh( rset->getFloat(3) );  
      dat.setRatioLow( rset->getFloat(4) );  
      dat.setRatioHigh( rset->getFloat(5) );  
      dat.setLUTValue( rset->getInt(6) );  
    
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrGroupDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigFgrGroupDat::writeArrayDB(const std::map< EcalLogicID, FEConfigFgrGroupDat >* data, FEConfigFgrInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigFgrGroupDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iconfid_vec= new int[nrows];
  int* xx= new int[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
  float* rr= new float[nrows];
  float* ss= new float[nrows];
  int* tt= new int[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* r_len= new ub2[nrows];
  ub2* s_len= new ub2[nrows];
  ub2* t_len= new ub2[nrows];


  // const EcalLogicID* channel;
  const FEConfigFgrGroupDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigFgrGroupDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        //channel = &(p->first);
	//	int logicID = channel->getLogicID();
	//	if (!logicID) { throw(std::runtime_error("FEConfigFgrGroupDat::writeArrayDB:  Bad EcalLogicID")); }
	//	ids[count]=logicID;
	iconfid_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iconf);
	int x=dataitem->getFgrGroupId();
	float y=dataitem->getThreshLow();
	float z=dataitem->getThreshHigh();
	float r=dataitem->getRatioLow();
	float s=dataitem->getRatioHigh();
	int t=dataitem->getLUTValue();

	xx[count]=x;
	yy[count]=y;
	zz[count]=z;
	rr[count]=r;
	ss[count]=s;
	tt[count]=t;

	//	ids_len[count]=sizeof(ids[count]);
	iconf_len[count]=sizeof(iconfid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	r_len[count]=sizeof(rr[count]);
	s_len[count]=sizeof(ss[count]);
	t_len[count]=sizeof(tt[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIFLOAT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)rr, OCCIFLOAT , sizeof(rr[0]), r_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)ss, OCCIFLOAT , sizeof(ss[0]), s_len );
    m_writeStmt->setDataBuffer(7, (dvoid*)tt, OCCIINT , sizeof(tt[0]), t_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iconfid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] rr;
    delete [] ss;
    delete [] tt;

    delete [] ids_len;
    delete [] iconf_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] r_len;
    delete [] s_len;
    delete [] t_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigFgrGroupDat::writeArrayDB():  "+e.getMessage()));
  }
}

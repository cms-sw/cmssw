#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigWeightGroupDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigWeightInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigWeightGroupDat::FEConfigWeightGroupDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_group_id=0;
  m_w0 = 0;
  m_w1 = 0;
  m_w2 = 0;
  m_w3 = 0;
  m_w4 = 0;

}



FEConfigWeightGroupDat::~FEConfigWeightGroupDat()
{
}



void FEConfigWeightGroupDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO fe_weight_per_group_dat (wei_conf_id, group_id, "
		      " w0, w1, w2, w3, w4 ) "
		      "VALUES (:wei_conf_id, :group_id, "
		      ":w0, :w1, :w2, :w3, :w4 )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightGroupDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigWeightGroupDat::writeDB(const EcalLogicID* ecid, const FEConfigWeightGroupDat* item, FEConfigWeightInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigWeightGroupDat::writeDB:  ICONF not in DB")); }
  /* no need for the logic id in this table 
     int logicID = ecid->getLogicID();
     if (!logicID) { throw(std::runtime_error("FEConfigWeightGroupDat::writeDB:  Bad EcalLogicID")); }
  */

  try {
    m_writeStmt->setInt(1, iconfID);

    m_writeStmt->setInt(2, item->getWeightGroupId());
    m_writeStmt->setFloat(3, item->getWeight0());
    m_writeStmt->setFloat(4, item->getWeight1());
    m_writeStmt->setFloat(5, item->getWeight2());
    m_writeStmt->setFloat(6, item->getWeight3());
    m_writeStmt->setFloat(7, item->getWeight4());

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightGroupDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigWeightGroupDat::fetchData(map< EcalLogicID, FEConfigWeightGroupDat >* fillMap, FEConfigWeightInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
     throw(std::runtime_error("FEConfigWeightGroupDat::fetchData:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT d.group_id, d.w0, d.w1, d.w2, d.w3, d.w4 "
		 "FROM fe_weight_per_group_dat d "
		 "WHERE wei_conf_id = :wei_conf_id order by d.group_id ");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, FEConfigWeightGroupDat > p;
    FEConfigWeightGroupDat dat;
    int ig=-1;
    while(rset->next()) {
      ig++;                          // we create a dummy logic_id
      p.first = EcalLogicID( "Group_id",     // name
			     ig );        // logic_id
			   
      dat.setWeightGroupId( rset->getInt(1) );  
      dat.setWeight0( rset->getFloat(2) );  
      dat.setWeight1( rset->getFloat(3) );  
      dat.setWeight2( rset->getFloat(4) );  
      dat.setWeight3( rset->getFloat(5) );  
      dat.setWeight4( rset->getFloat(6) );  
    
      p.second = dat;
      fillMap->insert(p);
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigWeightGroupDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigWeightGroupDat::writeArrayDB(const std::map< EcalLogicID, FEConfigWeightGroupDat >* data, FEConfigWeightInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigWeightGroupDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iconfid_vec= new int[nrows];
  int* xx= new int[nrows];
  float* yy= new float[nrows];
  float* zz= new float[nrows];
  float* rr= new float[nrows];
  float* ss= new float[nrows];
  float* tt= new float[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* r_len= new ub2[nrows];
  ub2* s_len= new ub2[nrows];
  ub2* t_len= new ub2[nrows];


  // const EcalLogicID* channel;
  const FEConfigWeightGroupDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigWeightGroupDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        // channel = &(p->first);
	//	int logicID = channel->getLogicID();
	//	if (!logicID) { throw(std::runtime_error("FEConfigWeightGroupDat::writeArrayDB:  Bad EcalLogicID")); }
	//	ids[count]=logicID;
	iconfid_vec[count]=iconfID;

	dataitem = &(p->second);
	// dataIface.writeDB( channel, dataitem, iconf);
	int x=dataitem->getWeightGroupId();
	float y=dataitem->getWeight0();
	float z=dataitem->getWeight1();
	float r=dataitem->getWeight2();
	float s=dataitem->getWeight3();
	float t=dataitem->getWeight4();

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
    m_writeStmt->setDataBuffer(7, (dvoid*)tt, OCCIFLOAT , sizeof(tt[0]), t_len );

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
    throw(std::runtime_error("FEConfigWeightGroupDat::writeArrayDB():  "+e.getMessage()));
  }
}

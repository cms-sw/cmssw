#ifndef ITIMINGDAT_H
#define ITIMINGDAT_H

#include <map>
#include <stdexcept>
#include <string>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "OnlineDB/Oracle/interface/Oracle.h"

class ITimingDat : public IDataItem {
 public:
  typedef oracle::occi::SQLException SQLException;
  typedef oracle::occi::ResultSet ResultSet;
  friend class EcalCondDBInterface;
 

ITimingDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_timingMean = 0;
  m_timingRMS = 0;
  m_taskStatus=0;
};



 ~ITimingDat(){};


  // User data methods
  inline std::string getTable() { return m_table_name;}
  inline void setTable(std::string x) { m_table_name=x; }

  inline void setTimingMean(float mean) { m_timingMean = mean; }
  inline float getTimingMean() const { return m_timingMean; }
  
  inline void setTimingRMS(float rms) { m_timingRMS = rms; }
  inline float getTimingRMS() const { return m_timingRMS; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }
  

 private:
void prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO " + getTable() + " (iov_id, logic_id, "
			"timing_mean, timing_rms , task_status ) "
			"VALUES (:iov_id, :logic_id, "
			":timing_mean, :timing_rms, :task_status )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ITimingDat::prepareWrite():  "+e.getMessage()));
  }
}


  template<class DATT, class IOVT>
  void writeDB(const EcalLogicID* ecid, const DATT* item, IOVT* iov)
    throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("ITimingDat::writeDB:  IOV not in DB")); }

  int logicID = ecid->getLogicID();
  if (!logicID) { throw(std::runtime_error("ITimingDat::writeDB:  Bad EcalLogicID")); }
  
  try {
    m_writeStmt->setInt(1, iovID);
    m_writeStmt->setInt(2, logicID);

    m_writeStmt->setFloat(3, item->getTimingMean() );
    m_writeStmt->setFloat(4, item->getTimingRMS() );
    m_writeStmt->setInt(5, item->getTaskStatus() );


    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ITimingDat::writeDB():  "+e.getMessage()));
  }
}

  template<class DATT, class IOVT>
    void writeArrayDB(const std::map< EcalLogicID, DATT >* data, IOVT* iov)
    throw(std::runtime_error)

{
  using oracle::occi::OCCIINT;
  using oracle::occi::OCCIFLOAT;

  this->checkConnection();
  this->checkPrepare();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("ITimingDat::writeArrayDB:  IOV not in DB")); }


  int nrows=data->size(); 
  int* ids= new int[nrows];
  int* iovid_vec= new int[nrows];
  float* xx= new float[nrows];
  float* yy= new float[nrows];
  int* st= new int[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* iov_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];


  const EcalLogicID* channel;
  const DATT* dataitem;
  int count=0;
  //  typedef std::map< EcalLogicID, DATT >::const_iterator CI;
  typedef typename std::map< EcalLogicID, DATT >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
        channel = &(p->first);
	int logicID = channel->getLogicID();
	if (!logicID) { throw(std::runtime_error("ITimingDat::writeArrayDB:  Bad EcalLogicID")); }
	ids[count]=logicID;
	iovid_vec[count]=iovID;

	dataitem = &(p->second);
	
	float x=dataitem->getTimingMean();
	float y=dataitem->getTimingRMS();
	int statu=dataitem->getTaskStatus();

	xx[count]=x;
	yy[count]=y;
	st[count]=statu;



	ids_len[count]=sizeof(ids[count]);
	iov_len[count]=sizeof(iovid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	st_len[count]=sizeof(st[count]);

	count++;
     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iovid_vec, OCCIINT, sizeof(iovid_vec[0]),iov_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)ids, OCCIINT, sizeof(ids[0]), ids_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)xx, OCCIFLOAT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)yy, OCCIFLOAT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );
 
    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] iovid_vec;
    delete [] xx;
    delete [] yy;
    delete [] st;

    delete [] ids_len;
    delete [] iov_len;
    delete [] x_len;
    delete [] y_len;
    delete [] st_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("ITimingDat::writeArrayDB():  "+e.getMessage()));
  }
}





  template<class DATT, class IOVT>
  void fetchData(std::map< EcalLogicID, DATT >* fillMap, IOVT* iov)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("ITimingDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {

    m_readStmt->setSQL("SELECT cv.name, cv.logic_id, cv.id1, cv.id2, cv.id3, cv.maps_to, "
		 "d.timing_mean, d.timing_rms, d.task_status "
		 "FROM channelview cv JOIN "+ getTable() +"  d "
		 "ON cv.logic_id = d.logic_id AND cv.name = cv.maps_to "
		 "WHERE d.iov_id = :iov_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    std::pair< EcalLogicID, DATT > p;
    DATT dat;
    while(rset->next()) {
      p.first = EcalLogicID( rset->getString(1),     // name
			     rset->getInt(2),        // logic_id
			     rset->getInt(3),        // id1
			     rset->getInt(4),        // id2
			     rset->getInt(5),        // id3
			     rset->getString(6));    // maps_to

      dat.setTimingMean( rset->getFloat(7) );
      dat.setTimingRMS( rset->getFloat(8) );
      dat.setTaskStatus( rset->getInt(9) );


      p.second = dat;
      fillMap->insert(p);
    }


  } catch (SQLException &e) {
    throw(std::runtime_error("ITimingDat::fetchData():  "+e.getMessage()));
  }
}






  // User data
  float m_timingMean;
  float m_timingRMS;
  bool m_taskStatus;
  std::string m_table_name ; 
   
};

#endif





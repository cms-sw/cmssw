#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODDelaysDat.h"

using namespace std;
using namespace oracle::occi;

ODDelaysDat::ODDelaysDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_sm = 0;
  m_fed = 0;
  m_tt = 0;
  m_t1 = 0;

}



ODDelaysDat::~ODDelaysDat()
{
}



void ODDelaysDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, sm_id, fed_id, tt_id, time_offset ) "
			"VALUES (:1, :2, :3, :4, :5 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDelaysDat::prepareWrite():  "+e.getMessage()));
  }
}



void ODDelaysDat::writeDB(const ODDelaysDat* item, ODFEDelaysInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getSMId());
    m_writeStmt->setInt(3, item->getFedId() );
    m_writeStmt->setInt(4, item->getTTId() );
    m_writeStmt->setInt(5, item->getTimeOffset() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDelaysDat::writeDB():  "+e.getMessage()));
  }
}



void ODDelaysDat::fetchData(std::vector< ODDelaysDat >* p, ODFEDelaysInfo* iov)
  throw(std::runtime_error)
{
  this->checkConnection();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("ODDelaysDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    m_readStmt->setSQL("SELECT * FROM " + getTable() + "WHERE rec_id = :rec_id order by sm_id, fed_id, tt_id ");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< ODDelaysDat > p;
    ODDelaysDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setSMId( rset->getInt(2) );
      dat.setFedId( rset->getInt(3) );
      dat.setTTId( rset->getInt(4) );
      dat.setTimeOffset( rset->getInt(5) );

      p->push_back( dat);

    }
  } catch (SQLException &e) {
    throw(std::runtime_error("ODDelaysDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void ODDelaysDat::writeArrayDB(const std::vector< ODDelaysDat > data, ODFEDelaysInfo* iov)
    throw(std::runtime_error)
{
  this->checkConnection();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("ODDelays::writeArrayDB:  ODFEDelaysInfo not in DB")); }


  int nrows=data.size(); 
  int* ids= new int[nrows];
  int* xx= new int[nrows];
  int* yy= new int[nrows];
  int* zz= new int[nrows];
  int* st= new int[nrows];



  ub2* ids_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];


  ODDelaysDat dataitem;
  
  int n_data= (int) data.size();

  for (int count = 0; count < n_data ; count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getSMId();
    yy[count]=dataitem.getFedId();
    zz[count]=dataitem.getTTId();
    st[count]=dataitem.getTimeOffset();


	ids_len[count]=sizeof(ids[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	st_len[count]=sizeof(st[count]);

     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)ids, OCCIINT, sizeof(ids[0]),ids_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] st;

    delete [] ids_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] st_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODDelaysDat::writeArrayDB():  "+e.getMessage()));
  }
}

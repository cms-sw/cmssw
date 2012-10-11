#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODBadTTDat.h"

using namespace std;
using namespace oracle::occi;

ODBadTTDat::ODBadTTDat()
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



ODBadTTDat::~ODBadTTDat()
{
}



void ODBadTTDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, tr_id, fed_id, tt_id, status ) "
			"VALUES (:1, :2, :3, :4, :5 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadTTDat::prepareWrite():  "+e.getMessage()));
  }
}



void ODBadTTDat::writeDB(const ODBadTTDat* item, ODBadTTInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getTRId());
    m_writeStmt->setInt(3, item->getFedId() );
    m_writeStmt->setInt(4, item->getTTId() );
    m_writeStmt->setInt(5, item->getStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadTTDat::writeDB():  "+e.getMessage()));
  }
}



void ODBadTTDat::fetchData(std::vector< ODBadTTDat >* p, ODBadTTInfo* iov)
  throw(std::runtime_error)
{
  this->checkConnection();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("ODBadTTDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    m_readStmt->setSQL("SELECT * FROM " + getTable() + " WHERE rec_id = :rec_id order by tr_id, fed_id, tt_id ");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< ODBadTTDat > p;
    ODBadTTDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setTRId( rset->getInt(2) );
      dat.setFedId( rset->getInt(3) );
      dat.setTTId( rset->getInt(4) );
      dat.setStatus( rset->getInt(5) );

      p->push_back( dat);

    }
  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadTTDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void ODBadTTDat::writeArrayDB(const std::vector< ODBadTTDat > data, ODBadTTInfo* iov)
    throw(std::runtime_error)
{
  this->checkConnection();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("ODDelays::writeArrayDB:  ODBadTTInfo not in DB")); }


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


  ODBadTTDat dataitem;
  

  for (size_t count = 0; count != data.size(); count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getTRId();
    yy[count]=dataitem.getFedId();
    zz[count]=dataitem.getTTId();
    st[count]=dataitem.getStatus();


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
    throw(std::runtime_error("ODBadTTDat::writeArrayDB():  "+e.getMessage()));
  }
}

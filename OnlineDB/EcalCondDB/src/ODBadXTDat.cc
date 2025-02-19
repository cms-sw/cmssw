#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODBadXTDat.h"

using namespace std;
using namespace oracle::occi;

ODBadXTDat::ODBadXTDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_sm = 0;
  m_fed = 0;
  m_tt = 0;
  m_xt = 0;
  m_t1 = 0;

}



ODBadXTDat::~ODBadXTDat()
{
}



void ODBadXTDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, sm_id, fed_id, tt_id, xt_id, status ) "
			"VALUES (:1, :2, :3, :4, :5 ,:6 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadXTDat::prepareWrite():  "+e.getMessage()));
  }
}



void ODBadXTDat::writeDB(const ODBadXTDat* item, ODBadXTInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getSMId());
    m_writeStmt->setInt(3, item->getFedId() );
    m_writeStmt->setInt(4, item->getTTId() );
    m_writeStmt->setInt(5, item->getXTId() );
    m_writeStmt->setInt(6, item->getStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadXTDat::writeDB():  "+e.getMessage()));
  }
}



void ODBadXTDat::fetchData(std::vector< ODBadXTDat >* p, ODBadXTInfo* iov)
  throw(std::runtime_error)
{
  this->checkConnection();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("ODBadXTDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    m_readStmt->setSQL("SELECT * FROM " + getTable() + "WHERE rec_id = :rec_id order by sm_id, fed_id, tt_id , xt_id ");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< ODBadXTDat > p;
    ODBadXTDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setSMId( rset->getInt(2) );
      dat.setFedId( rset->getInt(3) );
      dat.setTTId( rset->getInt(4) );
      dat.setXTId( rset->getInt(5) );
      dat.setStatus( rset->getInt(6) );

      p->push_back( dat);

    }
  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadXTDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void ODBadXTDat::writeArrayDB(const std::vector< ODBadXTDat > data, ODBadXTInfo* iov)
    throw(std::runtime_error)
{
  this->checkConnection();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("ODDelays::writeArrayDB:  ODBadXTInfo not in DB")); }


  int nrows=data.size(); 
  int* ids= new int[nrows];
  int* xx= new int[nrows];
  int* yy= new int[nrows];
  int* zz= new int[nrows];
  int* z1= new int[nrows];
  int* st= new int[nrows];



  ub2* ids_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* z1_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];


  ODBadXTDat dataitem;
  

  for (size_t count = 0; count != data.size(); count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getSMId();
    yy[count]=dataitem.getFedId();
    zz[count]=dataitem.getTTId();
    z1[count]=dataitem.getXTId();
    st[count]=dataitem.getStatus();


	ids_len[count]=sizeof(ids[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	z1_len[count]=sizeof(z1[count]);
	st_len[count]=sizeof(st[count]);

     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)ids, OCCIINT, sizeof(ids[0]),ids_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)z1, OCCIINT , sizeof(z1[0]), z1_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] z1;
    delete [] st;

    delete [] ids_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] z1_len;
    delete [] st_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODBadXTDat::writeArrayDB():  "+e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigBadTTDat.h"

using namespace std;
using namespace oracle::occi;

FEConfigBadTTDat::FEConfigBadTTDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_tcc = 0;
  m_fed = 0;
  m_tt = 0;
  m_t1 = 0;

}



FEConfigBadTTDat::~FEConfigBadTTDat()
{
}



void FEConfigBadTTDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, tcc_id, fed_id, tt_id, status ) "
			"VALUES (:1, :2, :3, :4, :5 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigBadTTDat::prepareWrite():  "+e.getMessage()));
  }
}



void FEConfigBadTTDat::writeDB(const FEConfigBadTTDat* item, FEConfigBadTTInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getTCCId() );
    m_writeStmt->setInt(3, item->getFedId());
    m_writeStmt->setInt(4, item->getTTId() );
    m_writeStmt->setInt(5, item->getStatus() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigBadTTDat::writeDB():  "+e.getMessage()));
  }
}



void FEConfigBadTTDat::fetchData(std::vector< FEConfigBadTTDat >* p, FEConfigBadTTInfo* iov)
  throw(std::runtime_error)
{
  this->checkConnection();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    //  throw(std::runtime_error("FEConfigBadTTDat::writeDB:  IOV not in DB")); 
    return;
  }

  try {
    m_readStmt->setSQL("SELECT * FROM " + getTable() + " WHERE rec_id = :rec_id order by tcc_id, fed_id, tt_id ");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< FEConfigBadTTDat > p;
    FEConfigBadTTDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setTCCId( rset->getInt(2) );
      dat.setFedId( rset->getInt(3) );
      dat.setTTId( rset->getInt(4) );
      dat.setStatus( rset->getInt(5) );

      p->push_back( dat);

    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigBadTTDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void FEConfigBadTTDat::writeArrayDB(const std::vector< FEConfigBadTTDat > data, FEConfigBadTTInfo* iov)
    throw(std::runtime_error)
{
  this->checkConnection();

  int iovID = iov->fetchID();
  if (!iovID) { throw(std::runtime_error("FEConfigDelays::writeArrayDB:  FEConfigBadTTInfo not in DB")); }


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


  FEConfigBadTTDat dataitem;
  

  for (size_t count = 0; count != data.size(); count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getTCCId();
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
    throw(std::runtime_error("FEConfigBadTTDat::writeArrayDB():  "+e.getMessage()));
  }
}

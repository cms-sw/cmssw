#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODGolBiasCurrentDat.h"

using namespace std;
using namespace oracle::occi;

ODGolBiasCurrentDat::ODGolBiasCurrentDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;


  m_gol = 0;
  m_fed = 0;
  m_tt = 0;
  m_cur = 0;
  m_pll_cur=0;
  m_sta=0;
}



ODGolBiasCurrentDat::~ODGolBiasCurrentDat()
{
}



void ODGolBiasCurrentDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, fed_id, tt_id, gol_id, gol_current, pll_current, status ) "
			"VALUES (:1, :2, :3, :4, :5 , :6, :7)");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentDat::prepareWrite():  "+e.getMessage()));
  }
}



void ODGolBiasCurrentDat::writeDB(const ODGolBiasCurrentDat* item, ODGolBiasCurrentInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getFedId() );
    m_writeStmt->setInt(3, item->getTTId() );
    m_writeStmt->setInt(4, item->getGolId() );
    m_writeStmt->setInt(5, item->getCurrent() );
    m_writeStmt->setInt(6, item->getPLLCurrent() );
    m_writeStmt->setInt(7, item->getStatus() );


    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentDat::writeDB():  "+e.getMessage()));
  }
}



void ODGolBiasCurrentDat::fetchData(std::vector< ODGolBiasCurrentDat >* p, ODGolBiasCurrentInfo* iov)
  throw(std::runtime_error)
{
  this->checkConnection();

  iov->setConnection(m_env, m_conn);
  int iovID = iov->fetchID();
  if (!iovID) { 
    std::cout <<"ID not in the DB"<< endl; 
    return;
  }

  try {
    m_readStmt->setSQL("SELECT * FROM " + getTable() + " WHERE rec_id = :rec_id order by fed_id, tt_id, gol_id ");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< ODGolBiasCurrentDat > p;
    ODGolBiasCurrentDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setFedId( rset->getInt(2) );
      dat.setTTId( rset->getInt(3) );
      dat.setGolId( rset->getInt(4) );
      dat.setCurrent( rset->getInt(5) );
      dat.setPLLCurrent( rset->getInt(6) );
      dat.setStatus( rset->getInt(7) );

      p->push_back( dat);

    }


  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void ODGolBiasCurrentDat::writeArrayDB(const std::vector< ODGolBiasCurrentDat > data, ODGolBiasCurrentInfo* iov)
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
  int* ww= new int[nrows];
  int* kk= new int[nrows];
  int* st= new int[nrows];



  ub2* ids_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* w_len= new ub2[nrows];
  ub2* k_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];

  ODGolBiasCurrentDat dataitem;
  

  for (int count = 0; count != (int)data.size(); count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getFedId();
    yy[count]=dataitem.getTTId();
    zz[count]=dataitem.getGolId();
    ww[count]=dataitem.getCurrent();
    kk[count]=dataitem.getPLLCurrent();
    st[count]=dataitem.getStatus();


	ids_len[count]=sizeof(ids[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	w_len[count]=sizeof(ww[count]);
	k_len[count]=sizeof(kk[count]);
	st_len[count]=sizeof(st[count]);


     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)ids, OCCIINT, sizeof(ids[0]),ids_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)ww, OCCIINT , sizeof(ww[0]), w_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)kk, OCCIINT , sizeof(kk[0]), k_len );
    m_writeStmt->setDataBuffer(7, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] ww;
    delete [] kk;
    delete [] st;

    delete [] ids_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] w_len;
    delete [] k_len;
    delete [] st_len;


  } catch (SQLException &e) {
    throw(std::runtime_error("ODGolBiasCurrentDat::writeArrayDB():  "+e.getMessage()));
  }
}

#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODWeightsDat.h"

using namespace std;
using namespace oracle::occi;

ODWeightsDat::ODWeightsDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_sm = 0;
  m_fed = 0;
  m_tt = 0;
  m_xt = 0;

}



ODWeightsDat::~ODWeightsDat()
{
}



void ODWeightsDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, sm_id, fed_id, tt_id, cry_id, wei0, wei1, wei2, wei3, wei4, wei5 ) "
			"VALUES (:1, :2, :3, :4, :5, :6, :7, :8 , :9, :10, :11 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsDat::prepareWrite():  "+e.getMessage()));
  }
}



void ODWeightsDat::writeDB(const ODWeightsDat* item, ODFEWeightsInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getSMId());
    m_writeStmt->setInt(3, item->getFedId() );
    m_writeStmt->setInt(4, item->getTTId() );
    m_writeStmt->setInt(5, item->getCrystalId() );
    
    m_writeStmt->setFloat(6, item->getWeight0() );
    m_writeStmt->setFloat(7, item->getWeight1() );
    m_writeStmt->setFloat(8, item->getWeight2() );
    m_writeStmt->setFloat(9, item->getWeight3() );
    m_writeStmt->setFloat(10, item->getWeight4() );
    m_writeStmt->setFloat(11, item->getWeight5() );

    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsDat::writeDB():  "+e.getMessage()));
  }
}



void ODWeightsDat::fetchData(std::vector< ODWeightsDat >* p, ODFEWeightsInfo* iov)
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
    m_readStmt->setSQL("SELECT * FROM " + getTable() + " WHERE rec_id = :rec_id order by sm_id, fed_id, tt_id, cry_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< ODWeightsDat > p;
    ODWeightsDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setSMId( rset->getInt(2) );
      dat.setFedId( rset->getInt(3) );
      dat.setTTId( rset->getInt(4) );
      dat.setCrystalId( rset->getInt(5) );
      dat.setWeight0( rset->getFloat(6) );
      dat.setWeight1( rset->getFloat(7) );
      dat.setWeight2( rset->getFloat(8) );
      dat.setWeight3( rset->getFloat(9) );
      dat.setWeight4( rset->getFloat(10) );
      dat.setWeight5( rset->getFloat(11) );

      p->push_back( dat);

    }


  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void ODWeightsDat::writeArrayDB(const std::vector< ODWeightsDat > data, ODFEWeightsInfo* iov)
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
  float* xx1= new float[nrows];
  float* yy1= new float[nrows];
  float* zz1= new float[nrows];
  float* xx2= new float[nrows];
  float* yy2= new float[nrows];
  float* zz2= new float[nrows];


  ub2* ids_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];
  ub2* st_len= new ub2[nrows];
  ub2* x1_len= new ub2[nrows];
  ub2* y1_len= new ub2[nrows];
  ub2* z1_len= new ub2[nrows];
  ub2* x2_len= new ub2[nrows];
  ub2* y2_len= new ub2[nrows];
  ub2* z2_len= new ub2[nrows];

  ODWeightsDat dataitem;
  
  int n_data= (int) data.size();
  for (int count = 0; count <n_data ; count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getSMId();
    yy[count]=dataitem.getFedId();
    zz[count]=dataitem.getTTId();
    st[count]=dataitem.getCrystalId();
    xx1[count]=dataitem.getWeight0();
    yy1[count]=dataitem.getWeight1();
    zz1[count]=dataitem.getWeight2();
    xx2[count]=dataitem.getWeight3();
    yy2[count]=dataitem.getWeight4();
    zz2[count]=dataitem.getWeight5();


	ids_len[count]=sizeof(ids[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);
	st_len[count]=sizeof(st[count]);
	x1_len[count]=sizeof(xx1[count]);
	y1_len[count]=sizeof(yy1[count]);
	z1_len[count]=sizeof(zz1[count]);
	x2_len[count]=sizeof(xx2[count]);
	y2_len[count]=sizeof(yy2[count]);
	z2_len[count]=sizeof(zz2[count]);

     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)ids, OCCIINT, sizeof(ids[0]),ids_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );
    m_writeStmt->setDataBuffer(5, (dvoid*)st, OCCIINT , sizeof(st[0]), st_len );
    m_writeStmt->setDataBuffer(6, (dvoid*)xx1, OCCIFLOAT , sizeof(xx1[0]), x1_len );
    m_writeStmt->setDataBuffer(7, (dvoid*)yy1, OCCIFLOAT , sizeof(yy1[0]), y1_len );
    m_writeStmt->setDataBuffer(8, (dvoid*)zz1, OCCIFLOAT , sizeof(zz1[0]), z1_len );
    m_writeStmt->setDataBuffer(9, (dvoid*)xx2, OCCIFLOAT , sizeof(xx2[0]), x2_len );
    m_writeStmt->setDataBuffer(10, (dvoid*)yy2, OCCIFLOAT , sizeof(yy2[0]), y2_len );
    m_writeStmt->setDataBuffer(11, (dvoid*)zz2, OCCIFLOAT , sizeof(zz2[0]), z2_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] xx;
    delete [] yy;
    delete [] zz;
    delete [] st;
    delete [] xx1;
    delete [] yy1;
    delete [] zz1;
    delete [] xx2;
    delete [] yy2;
    delete [] zz2;

    delete [] ids_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;
    delete [] st_len;
    delete [] x1_len;
    delete [] y1_len;
    delete [] z1_len;
    delete [] x2_len;
    delete [] y2_len;
    delete [] z2_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsDat::writeArrayDB():  "+e.getMessage()));
  }
}

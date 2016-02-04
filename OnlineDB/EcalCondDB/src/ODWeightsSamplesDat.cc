#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/ODWeightsSamplesDat.h"

using namespace std;
using namespace oracle::occi;

ODWeightsSamplesDat::ODWeightsSamplesDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_fed = 0;

}



ODWeightsSamplesDat::~ODWeightsSamplesDat()
{
}



void ODWeightsSamplesDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO "+getTable()+" (rec_id, fed_id, sample_id, weight_number ) "
			"VALUES (:1, :2, :3, :4 )");
  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsSamplesDat::prepareWrite():  "+e.getMessage()));
  }
}



void ODWeightsSamplesDat::writeDB(const ODWeightsSamplesDat* item, ODFEWeightsInfo* iov )
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt->setInt(1, item->getId());
    m_writeStmt->setInt(2, item->getFedId() );
    m_writeStmt->setInt(3, item->getSampleId() );
    m_writeStmt->setInt(4, item->getWeightNumber() );
    
    m_writeStmt->executeUpdate();
  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsSamplesDat::writeDB():  "+e.getMessage()));
  }
}



void ODWeightsSamplesDat::fetchData(std::vector< ODWeightsSamplesDat >* p, ODFEWeightsInfo* iov)
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
    m_readStmt->setSQL("SELECT * FROM " + getTable() + " WHERE rec_id = :rec_id order by fed_id, sample_id");
    m_readStmt->setInt(1, iovID);
    ResultSet* rset = m_readStmt->executeQuery();
    
    //    std::vector< ODWeightsSamplesDat > p;
    ODWeightsSamplesDat dat;
    while(rset->next()) {
      // dat.setId( rset->getInt(1) );
      dat.setFedId( rset->getInt(2) );
      dat.setSampleId( rset->getInt(3) );
      dat.setWeightNumber( rset->getInt(4) );

      p->push_back( dat);

    }

  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsSamplesDat::fetchData():  "+e.getMessage()));
  }
}

//  ************************************************************************   // 

void ODWeightsSamplesDat::writeArrayDB(const std::vector< ODWeightsSamplesDat > data, ODFEWeightsInfo* iov)
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

  ub2* ids_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];

  ODWeightsSamplesDat dataitem;
  
  int n_data= (int) data.size();
  for (int count = 0; count <n_data ; count++) {
    dataitem=data[count];
    ids[count]=iovID;
    xx[count]=dataitem.getFedId();
    yy[count]=dataitem.getSampleId();
    zz[count]=dataitem.getWeightNumber();


	ids_len[count]=sizeof(ids[count]);
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);

     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)ids, OCCIINT, sizeof(ids[0]),ids_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT , sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );
   

    m_writeStmt->executeArrayUpdate(nrows);

    delete [] ids;
    delete [] xx;
    delete [] yy;
    delete [] zz;

    delete [] ids_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;

  } catch (SQLException &e) {
    throw(std::runtime_error("ODWeightsSamplesDat::writeArrayDB():  "+e.getMessage()));
  }
}

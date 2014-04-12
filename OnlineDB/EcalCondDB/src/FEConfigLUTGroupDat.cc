#include <stdexcept>
#include <string>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/FEConfigLUTGroupDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"

using namespace std;
using namespace oracle::occi;

FEConfigLUTGroupDat::FEConfigLUTGroupDat()
{
  m_env = NULL;
  m_conn = NULL;
  m_writeStmt = NULL;
  m_readStmt = NULL;

  m_group_id=0;
  for(int i=0; i<1024; i++){
    m_lut[i] = 0;
  }

}



FEConfigLUTGroupDat::~FEConfigLUTGroupDat()
{
}



void FEConfigLUTGroupDat::prepareWrite()
  throw(std::runtime_error)
{
  this->checkConnection();

  try {
    m_writeStmt = m_conn->createStatement();
    m_writeStmt->setSQL("INSERT INTO fe_lut_per_group_dat (lut_conf_id, group_id, "
		      " lut_id, lut_value ) "
		      "VALUES (:lut_conf_id, :group_id, "
		      ":lut_id, :lut_value )" );
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTGroupDat::prepareWrite():  "+e.getMessage()));
  }
}

void FEConfigLUTGroupDat::writeDB(const EcalLogicID* ecid, const FEConfigLUTGroupDat* item, FEConfigLUTInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();

  cout<< "iconf="<< iconfID << endl;

  if (!iconfID) { throw(std::runtime_error("FEConfigLUTGroupDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=1024;
  int* iconfid_vec= new int[nrows];
  int* xx= new int[nrows];
  int* yy= new int[nrows];
  int* zz= new int[nrows];


  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];



    for(int count=0; count<nrows; count++){

	iconfid_vec[count]=iconfID;
	int x=item->getLUTGroupId();
	int y=count;
	int z=m_lut[count];

	xx[count]=x;
	yy[count]=y;
	zz[count]=z;

	iconf_len[count]=sizeof(iconfid_vec[count]);
	
	x_len[count]=sizeof(xx[count]);
	y_len[count]=sizeof(yy[count]);
	z_len[count]=sizeof(zz[count]);

     }


  try {
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );

    m_writeStmt->executeArrayUpdate(nrows);


    delete [] iconfid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;



    delete [] iconf_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;


  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTGroupDat::writeArrayDB():  "+e.getMessage()));
  }
}


void FEConfigLUTGroupDat::fetchData(map< EcalLogicID, FEConfigLUTGroupDat >* fillMap, FEConfigLUTInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  fillMap->clear();

  iconf->setConnection(m_env, m_conn);
  int iconfID = iconf->fetchID();
  if (!iconfID) { 
     throw(std::runtime_error("FEConfigLUTGroupDat::fetchData:  ICONF not in DB")); 
    return;
  }
  
  try {

    m_readStmt->setSQL("SELECT d.group_id, d.lut_id, d.lut_value "
		 "FROM fe_lut_per_group_dat d "
		 "WHERE lut_conf_id = :lut_conf_id order by d.group_id, d.lut_id ");
    m_readStmt->setInt(1, iconfID);
    ResultSet* rset = m_readStmt->executeQuery();

    FEConfigLUTGroupDat* dat(0);
    std::pair< EcalLogicID, FEConfigLUTGroupDat > p;


    int nrows=1024;

    int igold=-1;
    int ig=igold;

    while(rset->next()) {
      ig=rset->getInt(1);
      int il=rset->getInt(2);  
      int ival=rset->getInt(3);
      if(il==0){
	
	p.first = EcalLogicID( "Group_id",  ig );   // a dummy logic_id
	dat=new FEConfigLUTGroupDat();
	dat->setLUTGroupId( ig );
	dat->setLUTValue( il, ival );  
      } else {
	dat->setLUTValue( il, ival );
      }

      if(il==(nrows-1)){

	p.second = *dat;
	fillMap->insert(p);
	 delete dat;
      }
    }
  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTGroupDat::fetchData:  "+e.getMessage()));
  }
}

void FEConfigLUTGroupDat::writeArrayDB(const std::map< EcalLogicID, FEConfigLUTGroupDat >* data, FEConfigLUTInfo* iconf)
  throw(std::runtime_error)
{
  this->checkConnection();
  this->checkPrepare();

  int iconfID = iconf->fetchID();
  if (!iconfID) { throw(std::runtime_error("FEConfigLUTGroupDat::writeArrayDB:  ICONF not in DB")); }


  int nrows=data->size()*1024; 
  
  int* iconfid_vec= new int[nrows];
  int* xx= new int[nrows];
  int* yy= new int[nrows];
  int* zz= new int[nrows];



  ub2* iconf_len= new ub2[nrows];
  ub2* x_len= new ub2[nrows];
  ub2* y_len= new ub2[nrows];
  ub2* z_len= new ub2[nrows];


  const FEConfigLUTGroupDat* dataitem;
  int count=0;
  typedef map< EcalLogicID, FEConfigLUTGroupDat >::const_iterator CI;
  for (CI p = data->begin(); p != data->end(); ++p) {
	
	
	dataitem = &(p->second);
	int x=dataitem->getLUTGroupId();
	

	for (int i=0; i<1024; i++){
	  iconfid_vec[count]=iconfID;
	  int y=i;
	  int z=dataitem->getLUTValue(i);

	  xx[count]=x;
	  yy[count]=y;
	  zz[count]=z;
	  
	  
	  iconf_len[count]=sizeof(iconfid_vec[count]);
	  
	  x_len[count]=sizeof(xx[count]);
	  y_len[count]=sizeof(yy[count]);
	  z_len[count]=sizeof(zz[count]);
	  
	  count++;

	}
     }


  try {

    //    for (int i=0; i<nrows; i++){

    int i=0;
      cout << "about to insert "<< iconfid_vec[i]<<" " <<xx[i]<< " "<< yy[i]<< " "<< zz[i]<< endl;
     i=nrows-1;
      cout << "about to insert "<< iconfid_vec[i]<<" " <<xx[i]<< " "<< yy[i]<< " "<< zz[i]<< endl;
      // }
    m_writeStmt->setDataBuffer(1, (dvoid*)iconfid_vec, OCCIINT, sizeof(iconfid_vec[0]),iconf_len);
    m_writeStmt->setDataBuffer(2, (dvoid*)xx, OCCIINT, sizeof(xx[0]), x_len );
    m_writeStmt->setDataBuffer(3, (dvoid*)yy, OCCIINT , sizeof(yy[0]), y_len );
    m_writeStmt->setDataBuffer(4, (dvoid*)zz, OCCIINT , sizeof(zz[0]), z_len );

    m_writeStmt->executeArrayUpdate(nrows);


    delete [] iconfid_vec;
    delete [] xx;
    delete [] yy;
    delete [] zz;



    delete [] iconf_len;
    delete [] x_len;
    delete [] y_len;
    delete [] z_len;


  } catch (SQLException &e) {
    throw(std::runtime_error("FEConfigLUTGroupDat::writeArrayDB():  "+e.getMessage()));
  }
}

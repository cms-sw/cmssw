#ifndef COND_DBCommon_Ref_H
#define COND_DBCommon_Ref_H
#include <string>
// pool includes
#include "DataSvc/Ref.h"
#include "POOLCore/Exception.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/Placement.h"
// local includes
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Connection.h"
namespace cond{
  class Connection;
  /* 
     wrapper of pool::Ref smart pointer
  */
  template <typename T>
  class Ref{
  public:
    Ref():m_db(0),m_place(0){
    }
    Ref( cond::Connection& db, pool::Ref<T> ref ): 
      m_db(&db), m_data(ref), m_place(0) {
    }
    Ref( cond::Connection& db, const std::string& token ):
      m_db(&db),
      m_data( pool::Ref<T>(&(pooldb.DataSvc()), token) ),
      m_place(0){
    }
    Ref( cond::PoolStorageManager& pooldb, T* obj ):
      m_pooldb(&pooldb),
      m_data( pool::Ref<T>(&(pooldb.DataSvc()), obj) ),
      m_place(0){
    }
    virtual ~Ref(){
      if(m_place) delete m_place;
    }
    void markWrite( const std::string& containerName ) {
      try{
	if(!m_place){
	  m_place = new pool::Placement;
	  m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
	  m_place->setDatabase(m_pooldb->connectionString(), pool::DatabaseSpecification::PFN);
	  m_place->setContainerName(containerName);
	}
	m_data.markWrite(*m_place);
      }catch( const pool::Exception& er){
	throw cond::RefException("markWrite",er.what());
      }
    }
    void markUpdate(){
      try{
	m_data.markUpdate();
      }catch( const pool::Exception& er){
	throw cond::RefException("markUpdate",er.what());
      }
    }
    void markDelete(){
      try{
	m_data.markDelete();
      }catch( const pool::Exception& er){
	throw cond::RefException("markDelete",er.what());
      }
    }
    std::string token() const{
      return m_data.toString();
    }
    void clear( ){
      m_data.clear() ;
    }
    void reset( ){
      m_data.reset() ;
    }
    std::string containerName(){
      std::string contName=m_data.token()->contID();
      return contName;
    }
 
  };
}//ns cond
#endif
// COND_REF_H













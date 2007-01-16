#ifndef COND_DBCommon_Ref_H
#define COND_DBCommon_Ref_H
#include <string>
#include "DataSvc/Ref.h"
#include "POOLCore/Exception.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/Placement.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
namespace cond{
  class PoolStorageManager;
  /* 
     wrapper of pool::Ref smart pointer
  */
  template <typename T>
  class Ref{
  public:
    Ref():m_pooldb(0),m_place(0){
    }
    Ref( cond::PoolStorageManager& pooldb, pool::Ref<T> ref ): 
      m_pooldb(&pooldb), m_data(ref), m_place(0) {
    }
    Ref( cond::PoolStorageManager& pooldb, const std::string& token ):
      m_pooldb(&pooldb),
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
    //user does not have ownership
    T* ptr() const{
      T* result=0;
      try{
	result=m_data.ptr();
      }catch(const pool::Exception& er){
	throw cond::RefException("ptr",er.what());
      }
      return result;
    }
    T* operator->() const{
      return this->ptr();
    }
    T& operator * () const{
      try{
	//T& result=*m_data;
	return *m_data;
      }catch(const pool::Exception& er){
	throw cond::RefException( "operator * ",er.what() );
      }
    }
  private:
    cond::PoolStorageManager* m_pooldb;
    pool::Ref<T> m_data;
    pool::Placement* m_place;
  };
}//ns cond
#endif
// COND_REF_H













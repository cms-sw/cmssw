#ifndef COND_DBCommon_TypedRef_H
#define COND_DBCommon_TypedRef_H
#include <string>
// pool includes
#include "DataSvc/IDataSvc.h"
#include "DataSvc/Ref.h"
#include "POOLCore/Exception.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/Placement.h"
// local includes
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Connection.h"
namespace cond{
  class PoolTransaction;
  /** 
      typed Ref smart pointer
  */
  template <typename T>
    class TypedRef{
    public:
    // default constuctor
    TypedRef();
    // construct from pool ref
    TypedRef( cond::PoolTransaction& pooldb, 
	      pool::Ref<T> ref );
    // construct from token
    TypedRef( cond::PoolTransaction& pooldb, 
	      const std::string& token );
    // construct from object pointer, take ownership of the object
    TypedRef( cond::PoolTransaction& pooldb, 
	      T* obj );
    // copy constructor
    TypedRef( const TypedRef<T>& aCopy);
    // externalised token
    const std::string token() const;
    // object name
    std::string className() const;
    // container name
    std::string containerName() const;
    virtual ~TypedRef();
    // externalised token
    std::string token() const;
    // object name
    std::string className() const;
    // container name
    std::string containerName() const;
    /* update operations
    **/
    void markWrite( const std::string& containerName ); 
    void markUpdate();
    void markDelete();
    // return the real pointer
    T* ptr() const;
    // dereference operator
    T* operator->() const;
    // dereference operator
    T& operator*() const;
    // assignment operator
    TypedRef<T>& operator=(const TypedRef<T>&);
    private:
    pool::IDataSvc* m_datasvc;
    pool::Placement* m_place;
    // wrap pool Ref
    pool::Ref<T> m_data;
  };
}
//implementation

// default constuctor
template<typename T> 
cond::TypedRef::TypedRef():m_datasvc(0),m_place(0){  
}
template<typename T> 
cond::TypedRef::TypedRef(cond::PoolTransaction& pooldb, 
			 pool::Ref<T> ref):
  m_datasvc(&(pooldb.poolDataSvc())), m_data(ref){
  std::string con=transaction.parentConnection().connectStr();
  m_place = new pool::Placement;
  m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  m_place->setDatabase(con,pool::DatabaseSpecification::PFN);
}
// construct from token
template<typename T> 
cond::TypedRef::TypedRef( cond::PoolTransaction& pooldb, 
			  const std::string& token ):
  m_datasvc(&(pooldb.poolDataSvc())),m_data(m_datasvc, token){
  std::string con=transaction.parentConnection().connectStr();
  m_place = new pool::Placement;
  m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  m_place->setDatabase(con,pool::DatabaseSpecification::PFN);
}
// construct from object pointer, take ownership of the object
template<typename T> 
cond::TypedRef::TypedRef( cond::PoolTransaction& pooldb, T* obj ):m_datasvc(&(pooldb.poolDataSvc())),m_data(m_datasvc, obj){
  std::string con=transaction.parentConnection().connectStr();
  m_place = new pool::Placement;
  m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  m_place->setDatabase(con,pool::DatabaseSpecification::PFN);
}
// copy constructor???
template<typename T>
cond::TypedRef::TypedRef( const TypedRef<T>& aCopy){
}
template<typename T>
const std::string 
cond::TypedRef::token() const{
  return m_data.toString();
}
// object name
template<typename T> std::string 
cond::TypedRef::className() const{
  ROOT::Reflex::Type mytype=m_data.objectType();
  return mytype.Name();
}
template<typename T> std::string 
cond::TypedRef::containerName() const{
  return m_data.token()->contID();
}
// write
template<typename T> void 
cond::TypedRef::markWrite( const std::string& containerName ){
  try{
    m_place->setContainerName(containerName);
    m_data.markWrite(*m_place);
  }catch( const pool::Exception& er){
    throw cond::RefException("markWrite",er.what());
  }
}
template<typename T> void 
cond::TypedRef::markUpdate(){
  try{
    m_data.markUpdate();
  }catch( const pool::Exception& er){
    throw cond::RefException("markUpdate",er.what());
  }
}
template<typename T> void 
cond::TypedRef::markDelete(){
  try{
    m_data.markDelete();
  }catch( const pool::Exception& er){
    throw cond::RefException("markDelete",er.what());
  }
}
template<typename T> T* 
cond::TypedRef::ptr() const{
  T* result=0;
  try{
    result=m_data.ptr();
  }catch(const pool::Exception& er){
    throw cond::RefException("ptr",er.what());
  }
  return result;
}
template<typename T> T* 
cond::TypedRef::operator->() const{
  return this->ptr();
}
template<typename T> T& 
cond::TypedRef::operator*() const{
  try{
    return *m_data;
  }catch(const pool::Exception& er){
    throw cond::RefException( "operator * ",er.what() );
  }
}
#endif
// COND_TYPEDREF_H













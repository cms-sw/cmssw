#ifndef COND_DBCommon_TypedRef_H
#define COND_DBCommon_TypedRef_H
#include <string>
//#include <iostream>
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
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
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
    std::string token() const;
    // object name
    std::string className() const;
    // container name
    std::string containerName() const;
    ~TypedRef();
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
    std::string m_con;
    //pool::Placement* m_place;
    // wrap pool Ref
    pool::Ref<T> m_data;
  };
}
//implementation

// default constuctor
template<typename T> 
cond::TypedRef<T>::TypedRef():m_datasvc(0){  
}
template<typename T> 
cond::TypedRef<T>::TypedRef(cond::PoolTransaction& pooldb, 
			    pool::Ref<T> ref):
  m_datasvc(&(pooldb.poolDataSvc())),
  m_con(pooldb.parentConnection().connectStr()), 
  m_data(ref){
  //std::string con=pooldb.parentConnection().connectStr();
  //m_place =0;
  //m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  //m_place->setDatabase(con,pool::DatabaseSpecification::PFN);
}
// construct from token
template<typename T> 
cond::TypedRef<T>::TypedRef( cond::PoolTransaction& pooldb, 
			  const std::string& token ):
  m_datasvc(&(pooldb.poolDataSvc())),
  m_con(pooldb.parentConnection().connectStr()),
  m_data(m_datasvc, token){
  //m_place=0;
  //m_place = new pool::Placement;
  //m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  //m_place->setDatabase(con,pool::DatabaseSpecification::PFN);
}
// construct from object pointer, take ownership of the object
template<typename T> 
cond::TypedRef<T>::TypedRef( cond::PoolTransaction& pooldb, T* obj ):
  m_datasvc(&(pooldb.poolDataSvc())),
  m_con(pooldb.parentConnection().connectStr()),
  m_data(m_datasvc, obj){
  //m_place = new pool::Placement;
  //m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  //m_place->setDatabase(con,pool::DatabaseSpecification::PFN);
}
// copy constructor??? should copy metadata??
template<typename T>
cond::TypedRef<T>::TypedRef( const TypedRef<T>& aCopy){
  m_datasvc=aCopy.m_datasvc;
  m_con=aCopy.m_con;
  m_data=aCopy.m_data;
}
template<typename T> std::string 
cond::TypedRef<T>::token() const{
  return m_data.toString();
}
// object name
template<typename T> std::string 
cond::TypedRef<T>::className() const{
  Reflex::Type mytype=m_data.objectType();
  return mytype.Name();
}
template<typename T> std::string 
cond::TypedRef<T>::containerName() const{
  return m_data.token()->contID();
}
// write
template<typename T> void 
cond::TypedRef<T>::markWrite( const std::string& containerName ){
  pool::Placement place;
  place.setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  place.setDatabase(m_con,pool::DatabaseSpecification::PFN);
  place.setContainerName(containerName);
  try{
    m_data.markWrite(place);
  }catch( const pool::Exception& er){
    throw cond::RefException("markWrite",er.what());
  }
}
template<typename T> void 
cond::TypedRef<T>::markUpdate(){
  try{
    m_data.markUpdate();
  }catch( const pool::Exception& er){
    throw cond::RefException("markUpdate",er.what());
  }
}
template<typename T> void 
cond::TypedRef<T>::markDelete(){
  try{
    m_data.markDelete();
  }catch( const pool::Exception& er){
    throw cond::RefException("markDelete",er.what());
  }
}
template<typename T> T* 
cond::TypedRef<T>::ptr() const{
  T* result=0;
  try{
    result=m_data.ptr();    
  }catch(const pool::Exception& er){
    throw cond::RefException("ptr",er.what());
  }
  return result;
}
template<typename T> T* 
cond::TypedRef<T>::operator->() const{
  return this->ptr();
}
template<typename T> T& 
cond::TypedRef<T>::operator*() const{
  try{
    return *m_data;
  }catch(const pool::Exception& er){
    throw cond::RefException( "operator * ",er.what() );
  }
}
template<typename T> cond::TypedRef<T>& 
cond::TypedRef<T>::operator=(const cond::TypedRef<T>& aRef){
  m_datasvc=aRef.m_datasvc;
  m_con=aRef.m_con;
  m_data=aRef.m_data;
  return *this;  
}
template<typename T>
cond::TypedRef<T>::~TypedRef(){
}
#endif
// COND_TYPEDREF_H













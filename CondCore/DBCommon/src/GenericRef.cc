#include "CondCore/DBCommon/interface/GenericRef.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
//pool includes
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/Placement.h"
#include "DataSvc/AnyPtr.h"
#include "StorageSvc/DbReflex.h"

cond::GenericRef::GenericRef():m_datasvc(0),m_place(0){
}

cond::GenericRef::GenericRef( cond::PoolTransaction& transaction ):
  m_datasvc(&(transaction.poolDataSvc())),m_place(0){
  std::string con=transaction.parentConnection().connectStr();
  if( !transaction.isReadOnly() ){
    m_place=new pool::Placement;
    m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
    m_place->setDatabase(con, pool::DatabaseSpecification::PFN);
  }
}

cond::GenericRef::GenericRef( cond::PoolTransaction& pooldb, 
			      const std::string& tokenStr) :
  m_datasvc(&(pooldb.poolDataSvc())),m_place(0) {
  pool::Token token;
  const pool::Guid& classID=token.fromString(tokenStr).classID();
  pool::RefBase myobj(m_datasvc,tokenStr, pool::DbReflex::forGuid(classID).TypeInfo());
  m_data=myobj;
}


cond::GenericRef::GenericRef(cond::PoolTransaction& pooldb, 
			     const std::string& token,
			     const std::string& className
			     ):
  m_datasvc(&(pooldb.poolDataSvc())),m_place(0)
{
  const ROOT::Reflex::Type myclassType=ROOT::Reflex::Type::ByName(className);
  pool::RefBase myobj(m_datasvc,token,myclassType.TypeInfo() );
  m_data=myobj;
}
cond::GenericRef::GenericRef( cond::PoolTransaction& pooldb, 
			      const std::string& token,
			      const std::type_info& objType ):
  m_datasvc(&(pooldb.poolDataSvc())),m_place(0)
{
  pool::RefBase myobj(m_datasvc,token, objType);
  m_data=myobj;
}

const std::string 
cond::GenericRef::token() const{
  return m_data.toString();
}

std::string
cond::GenericRef::className() const{
  ROOT::Reflex::Type mytype=m_data.objectType();
  return mytype.Name();
}

std::string
cond::GenericRef::containerName() const{
  return m_data.token()->contID();  
}

void 
cond::GenericRef::markWrite(const std::string& container){
  m_place->setContainerName(container);
  m_data.markWrite(*m_place);
}

void 
cond::GenericRef::markUpdate(){
  m_data.markUpdate();
}

void 
cond::GenericRef::markDelete(){
  m_data.markDelete();
}
cond::GenericRef& 
cond::GenericRef::operator=(const cond::GenericRef& rhs){
  if(this == &rhs){
    return *this;
  }
  m_datasvc = rhs.m_datasvc;
  m_place = rhs.m_place;
  m_data = rhs.m_data;
  return *this;
}
/*
void 
cond::RefBase::copy(const cond::RefBase& source){
}
void 
cond::RefBase::copyDeep(const RefBase&){
}
void 
cond::RefBase::copyShallow(const RefBase&){
}
*/
std::string
cond::GenericRef::exportTo( cond::PoolTransaction& destdb ){
  if( destdb.isReadOnly() ){
    throw cond::Exception("cond::RefBase::exportTo destination transaction is readonly");
  }
  std::string containerName=this->containerName();
  std::string className=this->className();
  const ROOT::Reflex::Type myclassType=ROOT::Reflex::Type::ByName(className);
  pool::IDataSvc* destsvc=&(destdb.poolDataSvc());
  const pool::AnyPtr myPtr=m_data.object().get();
  pool::Placement destPlace;
  destPlace.setDatabase(destdb.parentConnection().connectStr(), 
			pool::DatabaseSpecification::PFN );
  destPlace.setContainerName(containerName);
  destPlace.setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  pool::RefBase mycopy(destsvc,myPtr,myclassType.TypeInfo());
  mycopy.markWrite(destPlace);
  //return token string of the copy
  return mycopy.toString();
}
void 
cond::GenericRef::clear(){  
}
void 
cond::GenericRef::reset(){
}
cond::GenericRef::~GenericRef(){
  if(m_place){
    delete m_place;
    m_place=0;
  }
}

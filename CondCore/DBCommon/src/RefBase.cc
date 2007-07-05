#include "CondCore/DBCommon/interface/RefBase.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/IConnectionProxy.h"
//pool includes
#include "POOLCore/Exception.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/Placement.h"
#include "DataSvc/AnyPtr.h"
cond::RefBase::RefBase():m_datasvc(0),m_place(0){
}
cond::RefBase::RefBase( cond::PoolTransaction& transaction ):m_datasvc(&(transaction.poolDataSvc())),m_place(0){
  std::string con=transaction.parentConnection().connectStr();
  if( !transaction.isReadOnly() ){
    m_place=new pool::Placement;
    m_place->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
    m_place->setDatabase(con, pool::DatabaseSpecification::PFN);
  }
}
cond::RefBase::RefBase(cond::PoolTransaction& pooldb, 
		       const std::string& token,
		       const std::string& className
		       ):
  m_datasvc(&(pooldb.poolDataSvc())),m_place(0)
{
  const ROOT::Reflex::Type myclassType=ROOT::Reflex::Type::ByName(className);
  pool::RefBase myobj(m_datasvc,token,myclassType.TypeInfo() );
  m_content=myobj;
}
cond::RefBase::RefBase( cond::PoolTransaction& pooldb, 
			const std::string& token,
			const std::type_info& objType ):
  m_datasvc(&(pooldb.poolDataSvc())),m_place(0)
{
  pool::RefBase myobj(m_datasvc,token, objType);
  m_content=myobj;
}
const std::string 
cond::RefBase::token() const{
  return m_content.toString();
}
std::string
cond::RefBase::className() const{
  ROOT::Reflex::Type mytype=m_content.objectType();
  return mytype.Name();
}
std::string
cond::RefBase::containerName() const{
  return m_content.token()->contID();  
}
void 
cond::RefBase::markWrite(const std::string& container){
  try{
    m_place->setContainerName(container);
    m_content.markWrite(*m_place);
  }catch( const pool::Exception& er){
    throw cond::RefException("markWrite",er.what());
  }
}
void 
cond::RefBase::markUpdate(){
  m_content.markUpdate();
}
void 
cond::RefBase::markDelete(){
  m_content.markDelete();
}
cond::RefBase& 
cond::RefBase::operator=(const cond::RefBase& rhs){
  if(this == &rhs){
    return *this;
  }
  m_datasvc = rhs.m_datasvc;
  m_place = rhs.m_place;
  m_content = rhs.m_content;
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
cond::RefBase::exportTo( cond::PoolTransaction& destdb ){
  if( destdb.isReadOnly() ){
    throw cond::Exception("cond::RefBase::exportTo destination transaction is readonly");
  }
  std::string containerName=this->containerName();
  std::string className=this->className();
  const ROOT::Reflex::Type myclassType=ROOT::Reflex::Type::ByName(className);
  pool::IDataSvc* destsvc=&(destdb.poolDataSvc());
  const pool::AnyPtr myPtr=m_content.object().get();
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
cond::RefBase::clear(){  
}
void 
cond::RefBase::reset(){
}
cond::RefBase::~RefBase(){
  if(m_place){
    delete m_place;
    m_place=0;
  }
}

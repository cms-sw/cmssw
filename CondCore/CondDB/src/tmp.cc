#include "CondCore/CondDB/interface/tmp.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"

namespace tmp {

bool Switch::open( const std::string& connectionString, bool readOnly ){
  cond::DbConnection oraConn;
  oraConn.configuration().setMessageLevel( coral::Debug );
  oraConn.configure();
  cond::DbSession oraSession = oraConn.createSession();
  //oraSession.open( connectionString, readOnly );
  oraSession.open( connectionString, cond::Auth::COND_READER_ROLE, readOnly );
  ora::Database& oraDb = oraSession.storage();
  new_impl::Session session;
  session.open( oraDb.storageAccessSession().share() );
  session.transaction().start();
  bool exists = session.existsDatabase();
  session.transaction().commit();
  if( !exists ){
    oraDb.transaction().start();
    exists = oraDb.exists();
    oraDb.transaction().commit();
    if( exists ){
      ORAImpl = oraSession;
      isORA = true; 
      return true;
    }
  }
  impl = session;
  return exists;
}
      
Transaction::Transaction( Switch& s ):
  m_switch( &s ){
}
Transaction::Transaction( const Transaction& rhs ):
  m_switch( rhs.m_switch ) {
}
Transaction& Transaction::operator=( const Transaction& rhs ){
  m_switch = rhs.m_switch ;
  return *this;
}

void Transaction::start( bool readOnly ){
  if( m_switch->isORA ){
    m_switch->ORAImpl.transaction().start( readOnly );
  } else {
    m_switch->impl.transaction().start( readOnly );
  }
}

void Transaction::commit(){
  if( m_switch->isORA ){
    m_switch->ORAImpl.transaction().commit();
  } else {
    m_switch->impl.transaction().commit();
  }
}

void Transaction::rollback(){
  if( m_switch->isORA ){
    m_switch->ORAImpl.transaction().rollback();
  } else {
    m_switch->impl.transaction().rollback();
  }
}

bool Transaction::isActive(){
  bool active = false;
  if( m_switch->isORA ){
    active = m_switch->ORAImpl.transaction().isActive();
  } else {
    active = m_switch->impl.transaction().isActive();
  }
  return active;
}

IOVProxy::Iterator::Iterator():
  m_impl(),
  m_ORAImpl(){
}

IOVProxy::Iterator::Iterator( new_impl::IOVProxy::Iterator impl ):
  m_impl( impl ),
  m_ORAImpl(){
}

IOVProxy::Iterator::Iterator( cond::IOVProxy::const_iterator impl ):
  m_impl(),
  m_ORAImpl( impl ),
  m_isORA( true ){
}

IOVProxy::Iterator::Iterator( const Iterator& rhs ):
  m_impl( rhs.m_impl ),
  m_ORAImpl( rhs.m_ORAImpl ),
  m_isORA( rhs.m_isORA ){
}

IOVProxy::Iterator& IOVProxy::Iterator::operator=( const Iterator& rhs ){
  if( this != &rhs ){
    m_impl = rhs.m_impl;
    m_ORAImpl = rhs.m_ORAImpl;
    m_isORA = rhs.m_isORA;
  }
  return *this;
}

conddb::Iov_t IOVProxy::Iterator::operator*() {
  conddb::Iov_t retVal;
  if( m_isORA ) {
    retVal.since = m_ORAImpl->since();
    retVal.till = m_ORAImpl->till();
    retVal.payloadId = m_ORAImpl->token();
  } else {
    retVal = *m_impl;
  }
  return retVal;
}

IOVProxy::Iterator& IOVProxy::Iterator::operator++(){
  if( m_isORA ) {
    m_ORAImpl.operator++();
  } else {
    m_impl.operator++();
  }
  return *this;
}

IOVProxy::Iterator IOVProxy::Iterator::operator++(int){
  Iterator ret;
  if( m_isORA ) {
    ret = Iterator( m_ORAImpl );
    m_ORAImpl.operator++();
  } else {
    ret = Iterator( m_impl );
    m_impl.operator++(0);
  }
  return ret;
}

bool IOVProxy::Iterator::operator==( const Iterator& rhs ) const {
  if( m_isORA != rhs.m_isORA ) return false;
  if( m_isORA ) {
    return m_ORAImpl == rhs.m_ORAImpl;
  } 
  return m_impl == rhs.m_impl;
}
      
bool IOVProxy::Iterator::operator!=( const Iterator& rhs ) const {
  return !operator==( rhs );
}

  IOVProxy::IOVProxy( const new_impl::IOVProxy& impl ):
  m_impl( impl ),
  m_ORAImpl(),
  m_ORATag(""){
}
IOVProxy::IOVProxy( cond::DbSession& ORASession ):
  m_impl(),
  m_ORAImpl( ORASession ),
  m_ORATag(""),
  m_isORA( true ){
}

IOVProxy::IOVProxy( const IOVProxy& rhs ):
  m_impl( rhs.m_impl ),
  m_ORAImpl( rhs.m_ORAImpl ),
  m_ORATag( rhs.m_ORATag ),
  m_isORA( rhs.m_isORA ){
}

      //
IOVProxy& IOVProxy::operator=( const IOVProxy& rhs ){
  m_impl = rhs.m_impl;
  m_ORAImpl = rhs.m_ORAImpl;
  m_ORATag = rhs.m_ORATag;
  m_isORA = rhs.m_isORA;
  return *this;
}

void IOVProxy::load( const std::string& tag, bool full ){
  if( m_isORA ){
    cond::MetaData metadata( m_ORAImpl.db() );
    m_ORAImpl.load( metadata.getToken(tag) );
    m_ORATag = tag;
  } else {
    m_impl.load( tag, full );
  }
}

void IOVProxy::reload(){
  if( m_isORA ){
    m_ORAImpl.refresh();
  } else {
    m_impl.reload();
  }
}

void IOVProxy::reset(){
  if( !m_isORA ){
    m_impl.reset();
  } 
  //else {
  //  not sure what to do in this case...
  //} 
}

std::string IOVProxy::tag() const {
  if( m_isORA ){
    return m_ORATag;
  } else {
    return m_impl.tag();
  } 
}

conddb::TimeType IOVProxy::timeType() const {
  if( m_isORA ){
    return (conddb::TimeType)m_ORAImpl.timetype();
  } else {
    return m_impl.timeType();
  } 
}

std::string IOVProxy::payloadObjectType() const {
  if( m_isORA ){
    std::set<std::string> types = m_ORAImpl.payloadClasses();
    return types.size()? *types.begin() : std::string("");
  } else {
    return m_impl.payloadObjectType();
  } 
}

conddb::Time_t IOVProxy::endOfValidity() const {
  if( m_isORA ){
    return m_ORAImpl.lastTill();
  } else {
    return m_impl.endOfValidity();
  } 
}
      
conddb::Time_t IOVProxy::lastValidatedTime() const {
  if( m_isORA ){
    return m_ORAImpl.tail(1).front().since();
  } else {
    return m_impl.lastValidatedTime();
  }
}

IOVProxy::Iterator IOVProxy::begin() const {
  if( m_isORA ){
    return Iterator( m_ORAImpl.begin() );
  } else {
    return Iterator( m_impl.begin() );
  }
}

IOVProxy::Iterator IOVProxy::end() const {
  if( m_isORA ){
    return Iterator( m_ORAImpl.end());
  } else {
    return Iterator( m_impl.end() );
  }
}

IOVProxy::Iterator IOVProxy::find(conddb::Time_t time){
  if( m_isORA ){
    return Iterator( m_ORAImpl.find( (cond::Time_t) time ) );
  } else {
    return Iterator( m_impl.find( time ) );
  }
}

conddb::Iov_t IOVProxy::getInterval( conddb::Time_t time ){
  if( m_isORA ){
    conddb::Iov_t ret;
    cond::IOVProxy::const_iterator valid = m_ORAImpl.find( (cond::Time_t) time );
    if( valid == m_ORAImpl.end() ){
      conddb::throwException( "Can't find a valid interval for the specified time.","IOVProxy::getInterval");
    }
    ret.since = valid->since();
    ret.till = valid->till();
    ret.payloadId = valid->token();
    return ret;
  } else {
    return m_impl.getInterval( time );
  }
}
    
int IOVProxy::size() const {
  if( m_isORA ){
    return m_ORAImpl.size();
  } else {
    return m_impl.size();
  }
}

IOVEditor::IOVEditor( const new_impl::IOVEditor& impl ):
  m_impl( impl ),
  m_ORAImpl( ),
  m_ORATag(""){
}

IOVEditor::IOVEditor( const cond::IOVEditor& ORAImpl ):
  m_impl(),
  m_ORAImpl( ORAImpl ),
  m_ORATag(""),
  m_isORA( true ){
}

IOVEditor::IOVEditor( const IOVEditor& rhs ):
  m_impl( rhs.m_impl ),
  m_ORAImpl( rhs.m_ORAImpl ),
  m_ORATag( rhs.m_ORATag ),
  m_isORA( rhs.m_isORA ){
}

IOVEditor& IOVEditor::operator=( const IOVEditor& rhs ){
  m_impl = rhs.m_impl;
  m_ORAImpl = rhs.m_ORAImpl;
  m_ORATag = rhs.m_ORATag;
  m_isORA = rhs.m_isORA;
  return *this;
}

void IOVEditor::load( const std::string& tag ){
  if( m_isORA ){
    cond::MetaData metadata( m_ORAImpl.proxy().db() );
    m_ORAImpl.load( metadata.getToken(tag) );
    m_ORATag = tag;
  } else {
    m_impl.load( tag );
  }  
}

std::string IOVEditor::tag() const {
  if( m_isORA ){
    return m_ORATag;
  } else {
    return m_impl.tag();
  }   
}

conddb::TimeType IOVEditor::timeType() const {
  if( m_isORA ){
    return (conddb::TimeType)m_ORAImpl.timetype();
  } else {
    return m_impl.timeType();
  }   
}
      
std::string IOVEditor::payloadType() const {
  if( m_isORA ){
    std::set<std::string> types = m_ORAImpl.proxy().payloadClasses();
    return types.size()? *types.begin() : std::string("");
  } else {
    return m_impl.payloadType();
  } 
}
      
conddb::SynchronizationType IOVEditor::synchronizationType() const {
  if( m_isORA ){
    return conddb::SYNCHRONIZATION_UNKNOWN;
  } else {
    return m_impl.synchronizationType();
  } 
}

conddb::Time_t IOVEditor::endOfValidity() const {
   if( m_isORA ){
     return m_ORAImpl.proxy().lastTill();
  } else {
     return m_impl.endOfValidity();
  } 
}

void IOVEditor::setEndOfValidity( conddb::Time_t validity ){
   if( m_isORA ){
     m_ORAImpl.updateClosure( validity );
  } else {
     m_impl.setEndOfValidity( validity );
  }   
}

std::string IOVEditor::description() const {
   if( m_isORA ){
     return m_ORAImpl.proxy().comment();
   } else {
     return m_impl.description();
   }
}
      
void IOVEditor::setDescription( const std::string& description ){
   if( m_isORA ){
     m_ORAImpl.stamp( description );
   } else {
     m_impl.setDescription( description );
   }
}
      
conddb::Time_t IOVEditor::lastValidatedTime() const {
   if( m_isORA ){
     return m_ORAImpl.proxy().tail(1).front().since();
   } else {
     return m_impl.lastValidatedTime();
   }  
}
      
void IOVEditor::setLastValidatedTime( conddb::Time_t time ){
  if(!m_isORA ){
    m_impl.setLastValidatedTime( time );
  }
}

void IOVEditor::insert( conddb::Time_t since, const conddb::Hash& payloadHash, bool checkType ){
   if( m_isORA ){
     m_ORAImpl.append( (cond::Time_t)since, payloadHash ); 
   } else {
     m_impl.insert( since, payloadHash, checkType );
   }  
}

void IOVEditor::insert( conddb::Time_t since, const conddb::Hash& payloadHash, 
				     const boost::posix_time::ptime& insertionTime, bool checkType ){
   if( m_isORA ){
     m_ORAImpl.append( (cond::Time_t)since, payloadHash ); 
   } else {
     m_impl.insert( since, payloadHash, insertionTime, checkType );
   }  
}

bool IOVEditor::flush(){
  if(!m_isORA ){
    return m_impl.flush();
  }
  return false;
}
      
bool IOVEditor::flush( const boost::posix_time::ptime& operationTime ){
  if(!m_isORA ){
    return m_impl.flush( operationTime );
  }
  return false;
}

GTProxy::Iterator::Iterator():
  m_impl(),
  m_ORAImpl(){
}
	
GTProxy::Iterator::Iterator( new_impl::GTProxy::Iterator impl ):
  m_impl( impl ),
  m_ORAImpl(){
}

GTProxy::Iterator::Iterator( std::set<cond::TagMetadata>::const_iterator impl ):
  m_impl(),
  m_ORAImpl( impl ),
  m_isORA( true ){
}
	
GTProxy::Iterator::Iterator( const Iterator& rhs ):
  m_impl( rhs.m_impl ),
  m_ORAImpl( rhs.m_ORAImpl ),
  m_isORA( rhs.m_isORA ){
}

GTProxy::Iterator& GTProxy::Iterator::operator=( const GTProxy::Iterator& rhs ){
  m_impl = rhs.m_impl;
  m_ORAImpl = rhs.m_ORAImpl;
  m_isORA = rhs.m_isORA;
  return *this;
}

conddb::GTEntry_t GTProxy::Iterator::operator*(){
  if( m_isORA ){
    return conddb::GTEntry_t( std::tie( m_ORAImpl->recordname, m_ORAImpl->labelname, m_ORAImpl->tag ) );
  } else {
    return *m_impl;
  }
}

GTProxy::Iterator& GTProxy::Iterator::operator++(){
  if( m_isORA ){
    m_ORAImpl.operator++();
  } else {
    m_impl.operator++();
  }
  return *this;
}
	
GTProxy::Iterator GTProxy::Iterator::operator++(int){
  Iterator ret;
  if( m_isORA ){
    ret = Iterator( m_ORAImpl );
    m_ORAImpl.operator++(0);
  } else {
    ret = Iterator( m_impl );
    m_impl.operator++(0);
  }
  return ret;
}

bool GTProxy::Iterator::operator==( const GTProxy::Iterator& rhs ) const {
  if( m_isORA != rhs.m_isORA ) return false;
  if( m_isORA ){
    return m_ORAImpl == rhs.m_ORAImpl;
  }
  return m_impl == rhs.m_impl;
}
	
bool GTProxy::Iterator::operator!=( const GTProxy:: Iterator& rhs ) const {
  return !this->operator==( rhs );
}
	
GTProxy::GTProxy( const new_impl::GTProxy& impl ):
  m_impl( impl ),
  m_ORASession(),
  m_ORAGTData(),
  m_ORAGTName(""){
} 

GTProxy::GTProxy( const cond::DbSession& ORASession ):
  m_impl(),
  m_ORASession( ORASession ),
  m_ORAGTData(),
  m_ORAGTName(""),
  m_isORA( true ){
} 
GTProxy::GTProxy( const GTProxy& rhs ):
  m_impl( rhs.m_impl ),
  m_ORASession( rhs.m_ORASession ),
  m_ORAGTData( rhs.m_ORAGTData ),
  m_ORAGTName( rhs.m_ORAGTName ),
  m_isORA( rhs.m_isORA ){
}
      
GTProxy& GTProxy::operator=( const GTProxy& rhs ){
  m_impl = rhs.m_impl;
  m_ORASession = rhs.m_ORASession;
  m_ORAGTData = rhs.m_ORAGTData;
  m_ORAGTName = rhs.m_ORAGTName;
  m_isORA = rhs.m_isORA;
  return *this;
}
            
void GTProxy::load( const std::string& gtName ){
  if( m_isORA ){
    cond::TagCollectionRetriever gtRetriever( m_ORASession );
    gtRetriever.getTagCollection( gtName, m_ORAGTData );
    m_ORAGTName = gtName;
  } else {
    m_impl.load( gtName );
  }
}

void GTProxy::reload(){
  if( m_isORA ){
    cond::TagCollectionRetriever gtRetriever( m_ORASession );
    gtRetriever.getTagCollection( m_ORAGTName, m_ORAGTData );
  } else {
    m_impl.reload();
  }
}

void GTProxy::reset(){
  if( m_isORA ){
    m_ORAGTData.clear();
  } else {
    m_impl.reset();
  }
}

std::string GTProxy::name() const {
  if( m_isORA ){
    return m_ORAGTName;
  } else {
    return m_impl.name();
  }
}

conddb::Time_t GTProxy::validity() const {
  if( m_isORA ){
    return conddb::time::MAX;
  } else {
    return m_impl.validity();
  }
}

boost::posix_time::ptime GTProxy::snapshotTime() const {
  if( m_isORA ){
    return boost::posix_time::ptime();
  } else {
    return m_impl.snapshotTime();
  }
}

GTProxy::Iterator GTProxy::begin() const {
  if( m_isORA ){
    return Iterator( m_ORAGTData.begin() );
  } else {
    return Iterator( m_impl.begin() );
  } 
}

GTProxy::Iterator GTProxy::end() const {
  if( m_isORA ){
    return Iterator( m_ORAGTData.end() );
  } else {
    return Iterator( m_impl.end() );
  } 
}
    
int GTProxy::size() const {
  if( m_isORA ){
    return m_ORAGTData.size();
  } else {
    return m_impl.size();
  } 
}

Session::Session():
  m_switch(),
  m_transaction( m_switch ){
}

// 
Session::Session( const Session& rhs ):
  m_switch( rhs.m_switch ),
  m_transaction( rhs.m_transaction ){
}

//
Session& Session::operator=( const Session& rhs ){
  m_switch = rhs.m_switch;
  m_transaction = rhs.m_transaction;
  return *this;
}

void Session::open( const std::string& connectionString, bool readOnly ){
  m_switch.open( connectionString, readOnly );
}

// 
void Session::close(){
  m_switch.ORAImpl.close();
  m_switch.impl.close();
}

//
conddb::Configuration& Session::configuration(){
  return m_switch.impl.configuration();
}

//
Transaction& Session::transaction(){
  return m_transaction;
}

IOVProxy Session::readIov( const std::string& tag, bool full ){
  if( m_switch.isORA ){
    IOVProxy proxy( m_switch.ORAImpl );
    proxy.load( tag, full );
    return proxy;
  } else {
    return IOVProxy( m_switch.impl.readIov( tag, full ) );
  }
}

// 
bool Session::existIov( const std::string& tag ){
  if( m_switch.isORA ){
    cond::MetaData metadata( m_switch.ORAImpl );
    return metadata.hasTag( tag );
  } else {
    return m_switch.impl.existsIov( tag );
  }
}

IOVEditor Session::createIov( const std::string& tag, conddb::TimeType timeType, const std::string& payloadType, 
							conddb::SynchronizationType synchronizationType ){
  if( m_switch.isORA ){
    cond::MetaData metadata( m_switch.ORAImpl );
    cond::IOVEditor ORAEditor( m_switch.ORAImpl );
    std::string tok = ORAEditor.create( (cond::TimeType )timeType );
    metadata.addMapping( tag, tok, (cond::TimeType )timeType );
    return IOVEditor( ORAEditor );
  } else {
    return IOVEditor( m_switch.impl.createIov( tag, timeType, payloadType, synchronizationType ) );
  }
}

IOVEditor Session::editIov( const std::string& tag ){
  if( m_switch.isORA ){
    cond::MetaData metadata( m_switch.ORAImpl );
    cond::IOVEditor ORAEditor( m_switch.ORAImpl );
    ORAEditor.load( metadata.getToken( tag ) );
    return IOVEditor( ORAEditor );
  } else {
    return IOVEditor( m_switch.impl.editIov( tag ) );
  }
}

IOVProxy Session::iovProxy(){
  if( m_switch.isORA ){
    IOVProxy proxy( m_switch.ORAImpl );
    return proxy;
  } else {
    return IOVProxy( m_switch.impl.iovProxy() );
  }
}

GTProxy Session::readGlobalTag( const std::string& name ){
  if( m_switch.isORA ){
    GTProxy gtReader( m_switch.ORAImpl  );
    gtReader.load( name );
    return gtReader;
  } else {
    return GTProxy( m_switch.impl.readGlobalTag( name ) );
  }  
}

}


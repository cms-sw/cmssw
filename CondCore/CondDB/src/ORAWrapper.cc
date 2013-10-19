#include "CondCore/CondDB/interface/ORAWrapper.h"

#include "DbCore.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"
#include "CondCore/TagCollection/interface/TagDBNames.h"

namespace cond {

  namespace ora_wrapper {

    Switch::Switch():
      oraImpl(),
      oraCurrent(),
      impl(){
    }

    Switch::Switch( const Switch& rhs ):
      oraImpl( rhs.oraImpl ),
      oraCurrent( rhs.oraCurrent ),
      impl( rhs.impl ){
    }
      
    Switch& Switch::operator=( const Switch& rhs ){
      oraImpl = oraImpl;
      oraCurrent = rhs.oraCurrent;
      impl = rhs.impl;
      return *this;
    }

    bool Switch::open( const std::string& connectionString, bool readOnly ){
      close();
      cond::DbConnection oraConnection;
      impl.configuration().configure( oraConnection.configuration() ); 
      oraConnection.configure();
      cond::DbSession oraSession = oraConnection.createSession();
      // fix me: what do we do for the roles?
      oraSession.open( connectionString, cond::Auth::COND_READER_ROLE, readOnly );
      auto oraDb = oraSession.storage();
      cond::persistency::Session session;
      session.open( oraDb.storageAccessSession().share() );
      session.transaction().start( true );
      bool exists = session.existsDatabase();
      session.transaction().commit();
      if( !exists ){
	oraDb.transaction().start( true );
	exists = oraDb.exists();
	if( !exists ){
	  exists = cond::persistency::existsTable( oraDb.storageAccessSession().share()->nominalSchema(), cond::tagInventoryTable.c_str() );
	}
	oraDb.transaction().commit();
	if( exists ){
	  oraCurrent = oraSession;
	  oraImpl.reset( new OraPool );
	  return true;
	}
      }
      impl = session;
      return exists;
    }
    
    void Switch::close(){
      if( oraImpl.get() ){
	oraImpl->clear();
      }
      oraImpl.reset();
      oraCurrent.close();
      impl.close();
    }
    
    bool Switch::isOra() const {
      return oraImpl.get();
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
      if( m_switch->isOra() ){
	m_switch->oraCurrent.transaction().start( readOnly );
      } else {
	m_switch->impl.transaction().start( readOnly );
      }
    }
    
    void Transaction::commit(){
      if( m_switch->isOra() ){
	m_switch->oraCurrent.transaction().commit();
      } else {
	m_switch->impl.transaction().commit();
      }
    }
    
    void Transaction::rollback(){
      if( m_switch->isOra() ){
	m_switch->oraCurrent.transaction().rollback();
      } else {
	m_switch->impl.transaction().rollback();
      }
    }
    
    bool Transaction::isActive(){
      bool active = false;
      if( m_switch->isOra() ){
	active = m_switch->oraCurrent.transaction().isActive();
      } else {
	active = m_switch->impl.transaction().isActive();
      }
      return active;
    }
    
    IOVProxy::Iterator::Iterator():
      m_impl(),
      m_oraImpl(){
    }
    
    IOVProxy::Iterator::Iterator( cond::persistency::IOVProxy::Iterator impl ):
      m_impl( impl ),
      m_oraImpl(){
    }
    
    IOVProxy::Iterator::Iterator( cond::IOVProxy::const_iterator impl ):
      m_impl(),
      m_oraImpl( impl ),
      m_isOra( true ){
    }
    
    IOVProxy::Iterator::Iterator( const Iterator& rhs ):
      m_impl( rhs.m_impl ),
      m_oraImpl( rhs.m_oraImpl ),
      m_isOra( rhs.m_isOra ){
    }
    
    IOVProxy::Iterator& IOVProxy::Iterator::operator=( const Iterator& rhs ){
      if( this != &rhs ){
	m_impl = rhs.m_impl;
	m_oraImpl = rhs.m_oraImpl;
	m_isOra = rhs.m_isOra;
      }
      return *this;
    }
    
    cond::Iov_t IOVProxy::Iterator::operator*() {
      cond::Iov_t retVal;
      if( m_isOra ) {
	retVal.since = m_oraImpl->since();
	retVal.till = m_oraImpl->till();
	retVal.payloadId = m_oraImpl->token();
      } else {
	retVal = *m_impl;
      }
      return retVal;
    }
    
    IOVProxy::Iterator& IOVProxy::Iterator::operator++(){
      if( m_isOra ) {
	m_oraImpl.operator++();
      } else {
	m_impl.operator++();
      }
      return *this;
    }
    
    IOVProxy::Iterator IOVProxy::Iterator::operator++(int){
      Iterator ret;
      if( m_isOra ) {
	ret = Iterator( m_oraImpl );
	m_oraImpl.operator++();
      } else {
	ret = Iterator( m_impl );
	m_impl.operator++(0);
      }
      return ret;
    }
    
    bool IOVProxy::Iterator::operator==( const Iterator& rhs ) const {
      if( m_isOra != rhs.m_isOra ) return false;
      if( m_isOra ) {
	return m_oraImpl == rhs.m_oraImpl;
      } 
      return m_impl == rhs.m_impl;
    }
    
    bool IOVProxy::Iterator::operator!=( const Iterator& rhs ) const {
      return !operator==( rhs );
    }
    
    IOVProxy::IOVProxy():
      m_impl(),
      m_oraImpl(),
      m_oraTag(){
    }
    
    IOVProxy::IOVProxy( const cond::persistency::IOVProxy& impl ):
      m_impl( impl ),
      m_oraImpl(),
      m_oraTag(){
    }
    IOVProxy::IOVProxy( cond::DbSession& ORASession ):
      m_impl(),
      m_oraImpl( ORASession ),
      m_oraTag( new std::string("") ){
    }
    
    IOVProxy::IOVProxy( const IOVProxy& rhs ):
      m_impl( rhs.m_impl ),
      m_oraImpl( rhs.m_oraImpl ),
      m_oraTag( rhs.m_oraTag ){
    }
    
    //
    IOVProxy& IOVProxy::operator=( const IOVProxy& rhs ){
      m_impl = rhs.m_impl;
      m_oraImpl = rhs.m_oraImpl;
      m_oraTag = rhs.m_oraTag;
      return *this;
    }
    
    void IOVProxy::load( const std::string& tag, bool full ){
      if( m_oraTag.get() ){
	cond::MetaData metadata( m_oraImpl.db() );
	m_oraImpl.load( metadata.getToken(tag) );
	*m_oraTag = tag;
      } else {
	m_impl.load( tag, full );
      }
    }
    
    void IOVProxy::reload(){
      if( m_oraTag.get() ){
	m_oraImpl.refresh();
      } else {
	m_impl.reload();
      }
    }
    
    void IOVProxy::reset(){
      if( !m_oraTag.get() ){
	m_impl.reset();
      } 
      //else {
      //  not sure what to do in this case...
      //} 
    }
    
    std::string IOVProxy::tag() const {
      if( m_oraTag.get() ){
	return *m_oraTag;
      } else {
	return m_impl.tag();
      } 
    }
    
    cond::TimeType IOVProxy::timeType() const {
      if( m_oraTag.get() ){
	return (cond::TimeType)m_oraImpl.timetype();
      } else {
	return m_impl.timeType();
      } 
    }
    
    std::string IOVProxy::payloadObjectType() const {
      if( m_oraTag.get() ){
	std::set<std::string> types = m_oraImpl.payloadClasses();
	return types.size()? *types.begin() : std::string("");
      } else {
	return m_impl.payloadObjectType();
      } 
    }
    
    cond::Time_t IOVProxy::endOfValidity() const {
      if( m_oraTag.get() ){
	return m_oraImpl.lastTill();
      } else {
	return m_impl.endOfValidity();
      } 
    }
    
    cond::Time_t IOVProxy::lastValidatedTime() const {
      if( m_oraTag.get() ){
	return m_oraImpl.tail(1).front().since();
      } else {
	return m_impl.lastValidatedTime();
      }
    }
    
    IOVProxy::Iterator IOVProxy::begin() const {
      if( m_oraTag.get() ){
	return Iterator( m_oraImpl.begin() );
      } else {
	return Iterator( m_impl.begin() );
      }
    }
    
    IOVProxy::Iterator IOVProxy::end() const {
      if( m_oraTag.get() ){
	return Iterator( m_oraImpl.end());
      } else {
	return Iterator( m_impl.end() );
      }
    }
    
    IOVProxy::Iterator IOVProxy::find(cond::Time_t time){
      if( m_oraTag.get() ){
	return Iterator( m_oraImpl.find( (cond::Time_t) time ) );
      } else {
	return Iterator( m_impl.find( time ) );
      }
    }
    
    cond::Iov_t IOVProxy::getInterval( cond::Time_t time ){
      if( m_oraTag.get() ){
	cond::Iov_t ret;
	cond::IOVProxy::const_iterator valid = m_oraImpl.find( (cond::Time_t) time );
	if( valid == m_oraImpl.end() ){
	  cond::throwException( "Can't find a valid interval for the specified time.","IOVProxy::getInterval");
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
      if( m_oraTag.get() ){
	return m_oraImpl.size();
      } else {
	return m_impl.size();
      }
    }
    
    IOVEditor::IOVEditor():
      m_impl(),
      m_oraImpl( ),
      m_oraTag(){
    }
    
    IOVEditor::IOVEditor( const cond::persistency::IOVEditor& impl ):
      m_impl( impl ),
      m_oraImpl( ),
      m_oraTag(){
    }
    
    IOVEditor::IOVEditor( const cond::IOVEditor& oraImpl ):
      m_impl(),
      m_oraImpl( oraImpl ),
      m_oraTag( new std::string ){
    }
    
    IOVEditor::IOVEditor( const IOVEditor& rhs ):
      m_impl( rhs.m_impl ),
      m_oraImpl( rhs.m_oraImpl ),
      m_oraTag( rhs.m_oraTag ){
    }
    
    IOVEditor& IOVEditor::operator=( const IOVEditor& rhs ){
      m_impl = rhs.m_impl;
      m_oraImpl = rhs.m_oraImpl;
      m_oraTag = rhs.m_oraTag;
      return *this;
    }
    
    void IOVEditor::load( const std::string& tag ){
      if( m_oraTag.get() ){
	cond::MetaData metadata( m_oraImpl.proxy().db() );
	m_oraImpl.load( metadata.getToken(tag) );
	*m_oraTag = tag;
      } else {
	m_impl.load( tag );
      }  
    }
    
    std::string IOVEditor::tag() const {
      if( m_oraTag.get() ){
	return *m_oraTag;
      } else {
	return m_impl.tag();
      }   
    }
    
    cond::TimeType IOVEditor::timeType() const {
      if( m_oraTag.get() ){
	return (cond::TimeType)m_oraImpl.timetype();
      } else {
	return m_impl.timeType();
      }   
    }
    
    std::string IOVEditor::payloadType() const {
      if( m_oraTag.get() ){
	std::set<std::string> types = m_oraImpl.proxy().payloadClasses();
	return types.size()? *types.begin() : std::string("");
      } else {
	return m_impl.payloadType();
      } 
    }
    
    cond::SynchronizationType IOVEditor::synchronizationType() const {
      if( m_oraTag.get() ){
	return cond::SYNCHRONIZATION_UNKNOWN;
      } else {
	return m_impl.synchronizationType();
      } 
    }
    
    cond::Time_t IOVEditor::endOfValidity() const {
      if( m_oraTag.get() ){
	return m_oraImpl.proxy().lastTill();
      } else {
	return m_impl.endOfValidity();
      } 
    }
    
    void IOVEditor::setEndOfValidity( cond::Time_t validity ){
      if( m_oraTag.get() ){
	m_oraImpl.updateClosure( validity );
      } else {
	m_impl.setEndOfValidity( validity );
      }   
    }
    
    std::string IOVEditor::description() const {
      if( m_oraTag.get() ){
	return m_oraImpl.proxy().comment();
      } else {
	return m_impl.description();
      }
    }
    
    void IOVEditor::setDescription( const std::string& description ){
      if( m_oraTag.get() ){
	m_oraImpl.stamp( description );
      } else {
	m_impl.setDescription( description );
      }
    }
    
    cond::Time_t IOVEditor::lastValidatedTime() const {
      if( m_oraTag.get() ){
	return m_oraImpl.proxy().tail(1).front().since();
      } else {
	return m_impl.lastValidatedTime();
      }  
    }
    
    void IOVEditor::setLastValidatedTime( cond::Time_t time ){
      if(!m_oraTag.get() ){
	m_impl.setLastValidatedTime( time );
      }
    }
    
    void IOVEditor::insert( cond::Time_t since, const cond::Hash& payloadHash, bool checkType ){
      if( m_oraTag.get() ){
	m_oraImpl.append( (cond::Time_t)since, payloadHash ); 
      } else {
	m_impl.insert( since, payloadHash, checkType );
      }  
    }
    
    void IOVEditor::insert( cond::Time_t since, const cond::Hash& payloadHash, 
			    const boost::posix_time::ptime& insertionTime, bool checkType ){
      if( m_oraTag.get() ){
	m_oraImpl.append( (cond::Time_t)since, payloadHash ); 
      } else {
	m_impl.insert( since, payloadHash, insertionTime, checkType );
      }  
    }

    bool IOVEditor::flush(){
      if(!m_oraTag.get() ){
	return m_impl.flush();
      }
      return true;
    }
    
    bool IOVEditor::flush( const boost::posix_time::ptime& operationTime ){
      if(!m_oraTag.get() ){
	return m_impl.flush( operationTime );
      }
      return true;
    }
    
    GTProxy::Iterator::Iterator():
      m_impl(),
      m_oraImpl(){
    }
    
    GTProxy::Iterator::Iterator( cond::persistency::GTProxy::Iterator impl ):
      m_impl( impl ),
      m_oraImpl(){
    }
    
    GTProxy::Iterator::Iterator( OraTagMap::const_iterator impl ):
      m_impl(),
      m_oraImpl( impl ),
      m_isOra( true ){
    }
    
    GTProxy::Iterator::Iterator( const Iterator& rhs ):
      m_impl( rhs.m_impl ),
      m_oraImpl( rhs.m_oraImpl ),
      m_isOra( rhs.m_isOra ){
    }
    
    GTProxy::Iterator& GTProxy::Iterator::operator=( const GTProxy::Iterator& rhs ){
      m_impl = rhs.m_impl;
      m_oraImpl = rhs.m_oraImpl;
      m_isOra = rhs.m_isOra;
      return *this;
    }
    
    cond::GTEntry_t GTProxy::Iterator::operator*(){
      if( m_isOra ){
	return cond::GTEntry_t( std::tie( m_oraImpl->second.recordname, m_oraImpl->second.labelname, m_oraImpl->second.tag ) );
      } else {
	return *m_impl;
      }
    }
    
    GTProxy::Iterator& GTProxy::Iterator::operator++(){
      if( m_isOra ){
	m_oraImpl.operator++();
      } else {
	m_impl.operator++();
      }
      return *this;
    }
    
    GTProxy::Iterator GTProxy::Iterator::operator++(int){
      Iterator ret;
      if( m_isOra ){
	ret = Iterator( m_oraImpl );
	m_oraImpl.operator++(0);
      } else {
	ret = Iterator( m_impl );
	m_impl.operator++(0);
      }
      return ret;
    }
    
    bool GTProxy::Iterator::operator==( const GTProxy::Iterator& rhs ) const {
      if( m_isOra != rhs.m_isOra ) return false;
      if( m_isOra ){
	return m_oraImpl == rhs.m_oraImpl;
      }
      return m_impl == rhs.m_impl;
    }
    
    bool GTProxy::Iterator::operator!=( const GTProxy:: Iterator& rhs ) const {
      return !this->operator==( rhs );
    }
    
    GTProxy::GTProxy():
      m_impl(),
      m_oraSession(),
      m_oraData(){
    } 
    
    GTProxy::GTProxy( const cond::persistency::GTProxy& impl ):
      m_impl( impl ),
      m_oraSession(),
      m_oraData(){
    } 
    
    GTProxy::GTProxy( const cond::DbSession& oraSession ):
      m_impl(),
      m_oraSession( oraSession ),
      m_oraData( new OraData ){
    } 
    
    GTProxy::GTProxy( const GTProxy& rhs ):
      m_impl( rhs.m_impl ),
      m_oraSession( rhs.m_oraSession ),
      m_oraData( rhs.m_oraData ){
    }
    
    GTProxy& GTProxy::operator=( const GTProxy& rhs ){
      m_impl = rhs.m_impl;
      m_oraSession = rhs.m_oraSession;
      m_oraData = rhs.m_oraData;
      return *this;
    }
    
    void GTProxy::load( const std::string& gtName ){
      if( m_oraData.get() ){
	std::set<cond::TagMetadata> tmp;
	cond::TagCollectionRetriever gtRetriever( m_oraSession, "", "" );
	gtRetriever.getTagCollection( gtName+"::All", tmp );
	for( auto m: tmp ){
	  std::string k = m.recordname;
	  if(!m.labelname.empty()) k+="_"+m.labelname;
	  m_oraData->gtData.insert( std::make_pair( k, m ) );
	}
	m_oraData->gt = gtName;
      } else {
	m_impl.load( gtName );
      }
    }
    
    void GTProxy::reload(){
      if( m_oraData.get() ){
	load( m_oraData->gt );
      } else {
	m_impl.reload();
      }
    }
    
    void GTProxy::reset(){
      if( m_oraData.get() ){
	m_oraData->gtData.clear();
      } else {
	m_impl.reset();
      }
    }
    
    std::string GTProxy::name() const {
      if( m_oraData.get() ){
	return m_oraData->gt;
      } else {
	return m_impl.name();
      }
    }
    
    cond::Time_t GTProxy::validity() const {
      if( m_oraData.get() ){
	return cond::time::MAX;
      } else {
	return m_impl.validity();
      }
    }
    
    boost::posix_time::ptime GTProxy::snapshotTime() const {
      if( m_oraData.get() ){
	return boost::posix_time::ptime();
      } else {
	return m_impl.snapshotTime();
      }
    }
    
    GTProxy::Iterator GTProxy::begin() const {
      if( m_oraData.get() ){
	return Iterator( m_oraData->gtData.begin() );
      } else {
	std::cout <<"# return new GTPRoxy begin... "<<std::endl;
	return Iterator( m_impl.begin() );
      } 
    }
    
    GTProxy::Iterator GTProxy::end() const {
      if( m_oraData.get() ){
	return Iterator( m_oraData->gtData.end() );
      } else {
	return Iterator( m_impl.end() );
      } 
    }
    
    int GTProxy::size() const {
      if( m_oraData.get() ){
	return m_oraData->gtData.size();
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
      m_switch.close();
    }
    
    //
    SessionConfiguration& Session::configuration(){
      return m_switch.impl.configuration();
    }
    
    //
    Transaction& Session::transaction(){
      return m_transaction;
    }
    
    bool Session::existsDatabase(){
      if( m_switch.isOra() ){
	return true;
      }
      return m_switch.impl.existsDatabase();
    }
    
    IOVProxy Session::readIov( const std::string& tag, bool full ){
      if( m_switch.isOra() ){
	IOVProxy proxy( m_switch.oraCurrent );
	proxy.load( tag, full );
	return proxy;
      } else {
	return IOVProxy( m_switch.impl.readIov( tag, full ) );
      }
    }
    
    // 
    bool Session::existIov( const std::string& tag ){
      if( m_switch.isOra() ){
	cond::MetaData metadata( m_switch.oraCurrent );
	return metadata.hasTag( tag );
      } else {
	return m_switch.impl.existsIov( tag );
      }
    }
    
    IOVEditor Session::createIov( const std::string& tag, cond::TimeType timeType, const std::string& payloadType, 
				  cond::SynchronizationType synchronizationType ){
      if( m_switch.isOra() ){
	cond::MetaData metadata( m_switch.oraCurrent );
	cond::IOVEditor oraEditor( m_switch.oraCurrent );
	std::string tok = oraEditor.create( (cond::TimeType )timeType );
	metadata.addMapping( tag, tok, (cond::TimeType )timeType );
	return IOVEditor( oraEditor );
      } else {
	return IOVEditor( m_switch.impl.createIov( tag, timeType, payloadType, synchronizationType ) );
      }
    }
    
    IOVEditor Session::editIov( const std::string& tag ){
      if( m_switch.isOra() ){
	cond::MetaData metadata( m_switch.oraCurrent );
	cond::IOVEditor oraEditor( m_switch.oraCurrent );
	oraEditor.load( metadata.getToken( tag ) );
	return IOVEditor( oraEditor );
      } else {
	return IOVEditor( m_switch.impl.editIov( tag ) );
      }
    }
    
    IOVProxy Session::iovProxy(){
      if( m_switch.isOra() ){
	IOVProxy proxy( m_switch.oraCurrent );
	return proxy;
      } else {
	return IOVProxy( m_switch.impl.iovProxy() );
      }
    }
    
    GTProxy Session::readGlobalTag( const std::string& name ){
      if( m_switch.isOra() ){
	GTProxy gtReader( m_switch.oraCurrent  );
	gtReader.load( name );
	return gtReader;
      } else {
	return GTProxy( m_switch.impl.readGlobalTag( name ) );
      }  
    }
    
    bool Session::isOra() const {
      return m_switch.isOra();
    }
    
  }
}


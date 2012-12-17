#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"

#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"

namespace cond {

  boost::shared_ptr<cond::IOVSequence> loadIOV( cond::DbSession& dbSess, 
						const std::string& iovToken ){
    if( iovToken.empty()){
      throw cond::Exception("IOVEditor::loadIOV Error: token is empty.");
    } 
    boost::shared_ptr<cond::IOVSequence> iov = dbSess.getTypedObject<cond::IOVSequence>( iovToken );
    // loading the lazy-loading Queryable vector...
    iov->loadAll();
    //**** temporary for the schema transition
    if( dbSess.isOldSchema() ){
      PoolTokenParser parser(  dbSess.storage() ); 
      iov->swapTokens( parser );
    }
    //****
    return iov;
  }
  
  std::string insertIOV( cond::DbSession& dbSess, 
			 const boost::shared_ptr<IOVSequence>& data,
			 bool swapOIds=false ){
    // ***** TEMPORARY FOR TRANSITION PHASE
    if( swapOIds && dbSess.isOldSchema() ){
      PoolTokenWriter writer( dbSess.storage() );
      data->swapOIds( writer );
    }
    // *****
    return dbSess.storeObject( data.get(), cond::IOVNames::container());
  }

  void updateIOV( cond::DbSession& dbSess, 
		  const boost::shared_ptr<IOVSequence>& data,
		  const std::string& token ){
    // ***** TEMPORARY FOR TRANSITION PHASE
    if( dbSess.isOldSchema() ){
      PoolTokenWriter writer( dbSess.storage() );
      data->swapOIds( writer );
    }
    // *****
    dbSess.updateObject( data.get(), token );
  }

  std::string EXPORT_REGISTRY_TABLE("COND_EXPORT_REGISTRY");

  ExportRegistry::ExportRegistry( DbConnection& conn ):
    m_conn( conn ){
    m_session = m_conn.createSession();
  }
  ExportRegistry::ExportRegistry():
    m_conn(){
    m_session = m_conn.createSession();
  }

  void ExportRegistry::open( const std::string& connectionString, 
			     bool readOnly ){
    m_session.open( connectionString, readOnly );
    cond::DbScopedTransaction trans( m_session );
    trans.start();
    coral::ISchema& schema = m_session.nominalSchema();
    if( !schema.existsTable( EXPORT_REGISTRY_TABLE ) ){
      coral::TableDescription descr( "CondDb" );
      descr.setName( EXPORT_REGISTRY_TABLE );
      descr.insertColumn( "OID",
			  coral::AttributeSpecification::typeNameForType<std::string>() );
      descr.setNotNullConstraint( "OID" );
      descr.setPrimaryKey( std::vector<std::string>( 1, "OID" ) );
      descr.insertColumn( "MAPPED_OID",
			  coral::AttributeSpecification::typeNameForType<std::string>() );
      descr.setNotNullConstraint( "MAPPED_OID" );
      coral::ITable& table = schema.createTable( descr );
      table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
    }
    trans.commit();
  
  }
  
  std::string ExportRegistry::lookup( const std::string& oId ){
    coral::ISchema& schema = m_session.nominalSchema();
    coral::ITable& table = schema.tableHandle( EXPORT_REGISTRY_TABLE );
    std::auto_ptr<coral::IQuery> query( table.newQuery() );
    query->addToOutputList( "MAPPED_OID" );
    query->defineOutputType( "MAPPED_OID", coral::AttributeSpecification::typeNameForType<std::string>());
    coral::AttributeList condData;
    condData.extend<std::string>("OID");
    condData[ "OID" ].data<std::string>() = oId;
    std::string condition = "OID =:OID";
    query->setCondition( condition, condData );
    coral::ICursor& cursor = query->execute();
    std::string ret("");
    if( cursor.next() ){
      ret = cursor.currentRow()["MAPPED_OID"].data<std::string>();
    }
    return ret;
  }

  void ExportRegistry::addMapping( const std::string& oId, 
				   const std::string& newOid ){
    cond::DbScopedTransaction trans( m_session );
    trans.start();
    std::string mapped = lookup( oId );
    if( !mapped.empty() ){
      throw cond::Exception("ExportRegistry::addMapping: specified oId:\""+oId+"\" has been mapped already.");
    }
    coral::ISchema& schema = m_session.nominalSchema();
    coral::ITable& table = schema.tableHandle( EXPORT_REGISTRY_TABLE );
    coral::AttributeList dataToInsert;
    dataToInsert.extend<std::string>( "OID");
    dataToInsert.extend<std::string>( "MAPPED_OID" );
    dataToInsert[ "OID" ].data<std::string>() = oId;
    dataToInsert[ "MAPPED_OID" ].data<std::string>() = newOid;
    table.dataEditor().insertRow( dataToInsert );
    trans.commit();
  }

  std::string ExportRegistry::getMapping( const std::string& oId ){
    std::string ret("");
    m_session.transaction().start( true );
    ret = lookup( oId );
    m_session.transaction().commit();
    return ret;
  }

  void ExportRegistry::close(){
    m_session.close();
  }

  IOVImportIterator::IOVImportIterator( boost::shared_ptr<cond::IOVProxyData>& destIov ):
    m_sourceIov(),
    m_destIov( destIov ),
    m_lastSince( 0 ),
    m_bulkSize( 0 ),
    m_cursor(),
    m_till(),
    m_registry( 0 )
  {
  }
  
  IOVImportIterator::~IOVImportIterator(){
  }

  void IOVImportIterator::setUp( cond::IOVProxy& sourceIov,
				 cond::Time_t since,
				 cond::Time_t till,
				 bool outOfOrder,
				 size_t bulkSize ){  
    m_sourceIov = sourceIov;
    const IOVSequence& siov = m_sourceIov.iov();
    cond::Time_t dsince = std::max(since, siov.firstSince());
    IOVSequence::const_iterator ifirstTill = siov.find(dsince);
    IOVSequence::const_iterator isecondTill = siov.find(till);
    if( isecondTill != siov.iovs().end() ) isecondTill++;
    
    if (ifirstTill==isecondTill) 
      throw cond::Exception("IOVImportIterator::setUp Error: empty input range");
    
    IOVSequence& diov = *m_destIov->data;
    if ( diov.iovs().empty()) ; // do not waist time
    else if (outOfOrder) {
      for( IOVSequence::const_iterator it=ifirstTill;
	   it!=isecondTill; ++it)
	if (diov.exist(it->sinceTime()))
	  throw cond::Exception("IOVImportIterator::setUp Error: since time already exists");
    } else if (dsince <= diov.iovs().back().sinceTime()) {
      std::ostringstream errStr;
      errStr << "IOVImportIterator::setUp Error: trying to append a since time " << dsince
	     << " which is not larger than last since " << diov.iovs().back().sinceTime();
      throw cond::Exception(errStr.str());
    }
    
    m_lastSince = dsince;
    m_cursor = ifirstTill;
    m_till = isecondTill;
    m_bulkSize = bulkSize;
 }

  void IOVImportIterator::setUp( cond::DbSession& sourceSess,
				 const std::string& sourceIovToken,
				 cond::Time_t since,
				 cond::Time_t till,
				 bool outOfOrder,
				 size_t bulkSize ){
    IOVProxy sourceIov( sourceSess, sourceIovToken );
    setUp( sourceIov, since, till, outOfOrder, bulkSize );
  }

  void IOVImportIterator::setUp( cond::IOVProxy& sourceIov,
				 size_t bulkSize ){ 

    m_sourceIov = sourceIov;
    const IOVSequence& siov = m_sourceIov.iov();
    cond::Time_t dsince = siov.firstSince();

    IOVSequence::const_iterator ifirstTill = siov.iovs().begin();
    IOVSequence::const_iterator isecondTill = siov.iovs().end();
    
    IOVSequence& diov = *m_destIov->data;
    if (!diov.iovs().empty()) { // do not waist time
      if (dsince <= diov.iovs().back().sinceTime()) {
	std::ostringstream errStr;
	errStr << "IOVImportIterator::setUp Error: trying to append a since time " << dsince
	       << " which is not larger than last since " << diov.iovs().back().sinceTime();
	throw cond::Exception(errStr.str());
      }
    }

    m_lastSince = dsince;
    m_cursor = ifirstTill;
    m_till = isecondTill;
    m_bulkSize = bulkSize;
  }

  void IOVImportIterator::setUp( cond::DbSession& sourceSess,
				 const std::string& sourceIovToken,
				 size_t bulkSize ){
    IOVProxy sourceIov( sourceSess, sourceIovToken );
    setUp( sourceIov, bulkSize );
  }

  void IOVImportIterator::setUp( cond::IOVProxy& sourceIov, 
				 cond::ExportRegistry& registry, 
				 size_t bulkSize ){
    m_registry = &registry;
    setUp( sourceIov, bulkSize );
  }

  bool IOVImportIterator::hasMoreElements(){
    return m_cursor != m_till;
  }

  std::string IOVImportIterator::importPayload( const std::string& payloadToken ){
    std::string newPayTok("");
    if( m_registry ){
      newPayTok = m_registry->getMapping( payloadToken );
      if(!newPayTok.empty() ) return newPayTok;
    }
    newPayTok = m_destIov->dbSession.importObject( m_sourceIov.db(),payloadToken );
    if( m_registry ) m_registry->addMapping( payloadToken, newPayTok );
    return newPayTok;
  }

  size_t IOVImportIterator::importMoreElements(){
    size_t i = 0;    
    boost::shared_ptr<IOVSequence>& diov = m_destIov->data;
    for( ; i<m_bulkSize && m_cursor != m_till; ++i, ++m_cursor, m_lastSince=m_cursor->sinceTime() ){
      std::string newPtoken = importPayload( m_cursor->token() );
      ora::OId poid;
      poid.fromString( newPtoken );
      ora::Container cont = m_destIov->dbSession.storage().containerHandle( poid.containerId() );
      diov->add(m_lastSince, newPtoken, cont.className());
    }
    if( m_cursor == m_till ) diov->stamp(cond::userInfo(),false);
    updateIOV( m_destIov->dbSession, diov, m_destIov->token );
    return i;
  }

  size_t IOVImportIterator::importAll(){
    size_t total = 0;
    while( hasMoreElements() ){
      total += importMoreElements();
    }
    return total;
  }
    
  IOVEditor::~IOVEditor(){}

  IOVEditor::IOVEditor( cond::DbSession& dbSess):
    m_isLoaded(false),
    m_iov( new IOVProxyData( dbSess ) ){
  }

  IOVEditor::IOVEditor( cond::DbSession& dbSess,
			const std::string& token ):
    m_isLoaded(false),
    m_iov( new IOVProxyData( dbSess, token )){
  }

  void IOVEditor::reload(){
    m_iov->refresh();
    m_isLoaded = true;
  }

  void IOVEditor::load( const std::string& token ){
    m_iov->token = token;
    m_iov->refresh();
    m_isLoaded = true;
  }

  void IOVEditor::debugInfo(std::ostream & co) const {
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    co << "IOVEditor: ";
    co << "db " << m_iov->dbSession.connectionString();
    if(m_iov->token.empty()) {
      co << " no token"; return;
    }
    if (!m_iov->data.get() )  {
      co << " no iov for token " << m_iov->token;
      return;
    }
    co << " iov token " << m_iov->token;
    co << "\nStamp: " <<  iov->comment()
       << "; time " <<  iov->timestamp()
       << "; revision " <<  iov->revision();
    co <<". TimeType " << cond::timeTypeSpecs[ iov->timeType()].name;
    if(  iov->iovs().empty() ) 
      co << ". empty";
    else
      co << ". size " <<  iov->iovs().size() 
	 << "; last since " << iov->iovs().back().sinceTime();
  }

  void IOVEditor::reportError(std::string message) const {
    std::ostringstream out;
    out << "Error in ";
    debugInfo(out);
    out  << "\n" << message;
    throw cond::Exception(out.str());
  }

  void IOVEditor::reportError( std::string message, 
                               cond::Time_t time) const {
    std::ostringstream out;
    out << "Error in";
    debugInfo(out);
    out << "\n" <<  message << " for time:  " << time;
    throw cond::Exception(out.str());
  }


  bool IOVEditor::createIOVContainerIfNecessary(){
    bool ret = false;
    cond::IOVSchemaUtility schemaUtil( m_iov->dbSession );
    if( !schemaUtil.existsIOVContainer() ){
      schemaUtil.createIOVContainer();
      ret = true;
    }
    return ret;
  }

  // create empty default sequence
  std::string IOVEditor::create( cond::TimeType timetype ) {
    m_iov->data.reset( new cond::IOVSequence(timetype) );
    m_iov->token = insertIOV( m_iov->dbSession, m_iov->data );
    m_isLoaded=true;
    return m_iov->token;
  }

  std::string IOVEditor::create(  cond::TimeType timetype,
				  cond::Time_t lastTill,
				  const std::string& metadata ) {
    m_iov->data.reset( new cond::IOVSequence((int)timetype,lastTill, metadata) );
    m_iov->token = insertIOV( m_iov->dbSession, m_iov->data );
    m_isLoaded=true;
    return m_iov->token;
  }
   
     // ####### TO BE REOMOVED ONLY USED IN TESTS
  std::string IOVEditor::create(cond::TimeType timetype, cond::Time_t lastTill ){
    m_iov->data.reset( new cond::IOVSequence((int)timetype,lastTill, std::string(" ")) );
    m_iov->token = insertIOV( m_iov->dbSession, m_iov->data );
    m_isLoaded=true;
    return m_iov->token;
  }

  bool IOVEditor::validTime( cond::Time_t time, 
                             cond::TimeType timetype) const {
    return time>=timeTypeSpecs[timetype].beginValue && time<=timeTypeSpecs[timetype].endValue;   
    
  }
  
  bool IOVEditor::validTime(cond::Time_t time) const {
    return validTime(time,m_iov->data->timeType());
  }
  
  
  
  unsigned int
  IOVEditor::insert( cond::Time_t tillTime,
		     const std::string& payloadToken ){
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    if( iov->iovs().empty() ) 
      reportError("cond::IOVEditor::insert cannot inser into empty IOV sequence",tillTime);
    
    if(!validTime(tillTime))
      reportError("cond::IOVEditor::insert time not in global range",tillTime);
    
    if(tillTime<=iov->lastTill() )
      reportError("cond::IOVEditor::insert IOV not in range",tillTime);
    
    cond::Time_t newSince=iov->lastTill()+1;
    std::string payloadClassName = m_iov->dbSession.classNameForItem( payloadToken );
    unsigned int ret = iov->add(newSince, payloadToken, payloadClassName);
    iov->updateLastTill(tillTime);
    updateIOV( m_iov->dbSession, iov, m_iov->token );
    return ret;
  }
  
  void 
  IOVEditor::bulkAppend(std::vector< std::pair<cond::Time_t, std::string > >& values){
    if (values.empty()) return;
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    cond::Time_t firstTime = values.front().first;
    cond::Time_t  lastTime = values.back().first;
    if(!validTime(firstTime))
      reportError("cond::IOVEditor::bulkInsert first time not in global range",firstTime);

    if(!validTime(lastTime))
      reportError("cond::IOVEditor::bulkInsert last time not in global range",lastTime);

    if(lastTime>= iov->lastTill() ||
      ( !iov->iovs().empty() && firstTime<=iov->iovs().back().sinceTime()) 
       )    
     reportError("cond::IOVEditor::bulkInsert IOV not in range",firstTime);

   for(std::vector< std::pair<cond::Time_t,std::string> >::const_iterator it=values.begin(); it!=values.end(); ++it){
     std::string payloadClassName = m_iov->dbSession.classNameForItem( it->second );    
     iov->add(it->first,it->second,payloadClassName );
   }
   updateIOV( m_iov->dbSession, iov, m_iov->token );
  }

  void 
  IOVEditor::bulkAppend(std::vector< cond::IOVElement >& values){
    if (values.empty()) return;
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    cond::Time_t firstTime = values.front().sinceTime();
    cond::Time_t   lastTime = values.back().sinceTime();
    if(!validTime(firstTime))
      reportError("cond::IOVEditor::bulkInsert first time not in global range",firstTime);

    if(!validTime(lastTime))
      reportError("cond::IOVEditor::bulkInsert last time not in global range",lastTime);

   if(lastTime>=iov->lastTill() ||
      ( !iov->iovs().empty() && firstTime<=iov->iovs().back().sinceTime()) 
      )    reportError("cond::IOVEditor::bulkInsert IOV not in range",firstTime);

   for(std::vector< cond::IOVElement >::const_iterator it=values.begin(); it!=values.end(); ++it){
     std::string payloadClassName = m_iov->dbSession.classNameForItem( it->token() );     
     iov->add(it->sinceTime(),it->token(),payloadClassName );
   }

   updateIOV( m_iov->dbSession, iov, m_iov->token );
  }

  void 
  IOVEditor::stamp( std::string const & icomment, 
                    bool append) {
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    iov->stamp(icomment, append);
    updateIOV( m_iov->dbSession, iov, m_iov->token );
  }

  void IOVEditor::editMetadata( std::string const & metadata, 
				bool append ){
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    iov->updateMetadata( metadata, append);
    updateIOV( m_iov->dbSession, iov, m_iov->token );
  }

  void IOVEditor::setScope( cond::IOVSequence::ScopeType scope ){
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    iov->setScope( scope );
    updateIOV( m_iov->dbSession, iov, m_iov->token );
  }

  void 
  IOVEditor::updateClosure( cond::Time_t newtillTime ){
    if( m_iov->token.empty() ) reportError("cond::IOVEditor::updateClosure cannot change non-existing IOV index");
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    iov->updateLastTill(newtillTime);
    updateIOV( m_iov->dbSession, iov, m_iov->token );
  }
  
  unsigned int 
  IOVEditor::append( cond::Time_t sinceTime,
		     const std::string& payloadToken ){
    if( m_iov->token.empty() ) {
      reportError("cond::IOVEditor::appendIOV cannot append to non-existing IOV index");
    }
    
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;

    if(!validTime(sinceTime))
      reportError("cond::IOVEditor::append time not in global range",sinceTime);  
    
    if(  !iov->iovs().empty() ){
      //range check in case 
      cond::Time_t lastValidSince=iov->iovs().back().sinceTime();
      if( sinceTime<= lastValidSince){
	std::ostringstream errStr;
	errStr << "IOVEditor::append Error: trying to append a since time " << lastValidSince
	       << " which is not larger than last since";
	reportError(errStr.str(), sinceTime);
      }
    }

    // does it make sense? (in case of mixed till and since insertions...)
    if (iov->lastTill()<=sinceTime) iov->updateLastTill( timeTypeSpecs[iov->timeType()].endValue );
    std::string payloadClassName = m_iov->dbSession.classNameForItem( payloadToken );   
    unsigned int ret = iov->add(sinceTime,payloadToken, payloadClassName );
    updateIOV( m_iov->dbSession, iov, m_iov->token );
    return ret;
  }

 
  unsigned int 
 IOVEditor::freeInsert( cond::Time_t sinceTime ,
			const std::string& payloadToken ){
    if( m_iov->token.empty() ) {
      reportError("cond::IOVEditor::freeInsert cannot append to non-existing IOV index");
    }
    
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    
    //   if( m_iov->data->iov.empty() ) reportError("cond::IOVEditor::freeInsert cannot insert  to empty IOV index");
    

   if(!validTime(sinceTime))
     reportError("cond::IOVEditor::freeInsert time not in global range",sinceTime);

   
   // we do not support multiple iov with identical since...
   if (m_iov->data->exist(sinceTime))
     reportError("cond::IOVEditor::freeInsert sinceTime already existing",sinceTime);



     // does it make sense? (in case of mixed till and since insertions...)
   if (iov->lastTill()<sinceTime) iov->updateLastTill( timeTypeSpecs[iov->timeType()].endValue );
   std::string payloadClassName = m_iov->dbSession.classNameForItem( payloadToken );   
   unsigned int ret = iov->add(sinceTime,payloadToken, payloadClassName );
   updateIOV( m_iov->dbSession, iov, m_iov->token );
   return ret;
  }


  // remove last entry
  unsigned int IOVEditor::truncate(bool withPayload) {
    if( m_iov->token.empty() ) reportError("cond::IOVEditor::truncate cannot delete to non-existing IOV sequence");
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    if (iov->piovs().empty()) return 0;
    if(withPayload){
      std::string tokenStr = iov->piovs().back().token();
      m_iov->dbSession.deleteObject( tokenStr );
    }
    unsigned int ret = iov->truncate();
    updateIOV( m_iov->dbSession, iov, m_iov->token );
    return ret;
    
  }


  void 
  IOVEditor::deleteEntries(bool withPayload){
    if( m_iov->token.empty() ) reportError("cond::IOVEditor::deleteEntries cannot delete to non-existing IOV sequence");
    if( !m_isLoaded ){
      reload();
    }
    boost::shared_ptr<IOVSequence>& iov = m_iov->data;
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      IOVSequence::const_iterator payloadItEnd=iov->piovs().end();
      for(payloadIt=iov->piovs().begin();payloadIt!=payloadItEnd;++payloadIt){
        tokenStr=payloadIt->token();
        m_iov->dbSession.deleteObject( tokenStr );
      }
    }
    m_iov->dbSession.deleteObject( m_iov->token );
    iov->piovs().clear();
  }

  size_t IOVEditor::import( cond::DbSession& sourceSess, 
			    const std::string& sourceIovToken ){
    boost::shared_ptr<IOVImportIterator> importer = importIterator();
    importer->setUp( sourceSess, sourceIovToken );
    return importer->importAll();
  }
    
  boost::shared_ptr<IOVImportIterator> 
  IOVEditor::importIterator(){
    if( !m_isLoaded ){
      reload();
    }
    return boost::shared_ptr<IOVImportIterator>( new IOVImportIterator( m_iov ));						 
  }

  TimeType IOVEditor::timetype() const {
    return m_iov->data->timeType();
  }

  std::string const & IOVEditor::token() const {
    return m_iov->token;
  }

  IOVProxy IOVEditor::proxy(){
    return IOVProxy( m_iov );
  }

}

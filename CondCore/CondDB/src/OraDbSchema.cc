#include "CondCore/CondDB/interface/Exception.h"
#include "OraDbSchema.h"
//
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"
#include "CondCore/TagCollection/interface/TagDBNames.h"
//
// externals
#include "RelationalAccess/ISchema.h"

namespace cond {

  namespace persistency {

    IOVCache::IOVCache( cond::DbSession& s ):
      m_iovAccess( s ){
    }
      
    cond::DbSession& IOVCache::session(){
      return m_iovAccess.proxy().db();
    }
      
    cond::IOVProxy IOVCache::iovSequence(){
      return m_iovAccess.proxy();
    }

    cond::IOVEditor IOVCache::editor(){
      return m_iovAccess;
    }
  
    bool IOVCache::existsTag( const std::string& t ){
      cond::MetaData metadata( session() );
      return metadata.hasTag( t );
    }

    std::string IOVCache::getToken( const std::string& tag ){
      if( tag != m_tag ){
	cond::MetaData metadata( session() );
	m_tag = tag;
	m_token = metadata.getToken( tag );
      }
      return m_token;
    }
      
    void IOVCache::addTag( const std::string& tag, const std::string token ){
      if( tag != m_tag ){
	cond::MetaData metadata( session() );
	metadata.addMapping( tag, token );
	m_tag = tag;
	m_token = token;
      }
    }

    bool IOVCache::load( const std::string& tag ){
      std::string token = getToken( tag );
      if( token.empty() ) return false;
      if(m_iovAccess.token() != token) m_iovAccess.load( token );
      return true;
    }

    OraTagTable::OraTagTable( IOVCache& cache ):
      m_cache( cache ){
    }

    bool OraTagTable::select( const std::string& name ){
      return m_cache.existsTag( name );
    }
      
    bool OraTagTable::select( const std::string& name, cond::TimeType& timeType, std::string& objectType, 
			      cond::SynchronizationType&, cond::Time_t& endOfValidity, 
			      std::string& description, cond::Time_t& lastValidatedTime ){
      if(!m_cache.load( name )) return false;
      timeType = m_cache.iovSequence().timetype();
      if( m_cache.iovSequence().payloadClasses().size()==0 ) throwException( "No payload type information found.","OraTagTable::select");
      objectType = *m_cache.iovSequence().payloadClasses().begin();
      endOfValidity = m_cache.iovSequence().lastTill();
      description = m_cache.iovSequence().comment();
      lastValidatedTime = m_cache.iovSequence().tail(1).back().since();
      return true;
    }

    bool OraTagTable::getMetadata( const std::string& name, std::string& description, 
				   boost::posix_time::ptime&, boost::posix_time::ptime& ){
      if(!m_cache.load( name )) return false;
      description = m_cache.iovSequence().comment();
      // TO DO: get insertion / modification time from the Logger?      
      return true;
    }
      
    void OraTagTable::insert( const std::string& name, cond::TimeType timeType, const std::string&, 
			      cond::SynchronizationType, cond::Time_t endOfValidity, const std::string& description, 
			      cond::Time_t, const boost::posix_time::ptime& ){
      std::string tok = m_cache.editor().create( timeType, endOfValidity );
      if( !m_cache.validationMode() ){
	m_cache.editor().stamp( description );
      }
      m_cache.addTag( name, tok );
    }
      
    void OraTagTable::update( const std::string& name, cond::Time_t& endOfValidity, const std::string& description, 
			      cond::Time_t, const boost::posix_time::ptime&  ){
      
      std::string tok = m_cache.getToken( name );
      if( tok.empty() ) throwException( "Tag \""+name+"\" has not been found in the database.","OraTagTable::update");
      m_cache.editor().load( tok );
      m_cache.editor().updateClosure( endOfValidity );
      if( !m_cache.validationMode() ) m_cache.editor().stamp( description );
    }
      
    void OraTagTable::updateValidity( const std::string&, cond::Time_t, 
				      const boost::posix_time::ptime& ){
      // can't be done in this case...
    }

    void OraTagTable::setValidationMode(){
      m_cache.setValidationMode();
    }

    OraPayloadTable::OraPayloadTable( DbSession& session ):
      m_session( session ){
    }

    bool OraPayloadTable::select( const cond::Hash& payloadHash, 
				  std::string& objectType, 
				  cond::Binary& payloadData, 
				  cond::Binary& //streamerInfoData
				){
      ora::Object obj = m_session.getObject( payloadHash );
      objectType = obj.typeName();
      payloadData.fromOraObject(obj );
      return true;
    }

    bool OraPayloadTable::getType( const cond::Hash& payloadHash, std::string& objectType ){
      objectType = m_session.classNameForItem( payloadHash );
      return true;
    }
      
    cond::Hash OraPayloadTable::insertIfNew( const std::string& objectType, 
					     const cond::Binary& payloadData, 
					     const cond::Binary&, //streamerInfoData
					     const boost::posix_time::ptime& ){
      ora::Object obj = payloadData.oraObject();
      std::string tok = m_session.storeObject( obj, objectType );
      m_session.flush();
      return tok;
    }

    OraIOVTable::OraIOVTable( IOVCache& iovCache ):
      m_cache( iovCache ){
    }

    size_t OraIOVTable::selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups ){
      if(!m_cache.load( tag )) return 0;
      if( m_cache.iovSequence().size()>0 ){
	groups.push_back( m_cache.iovSequence().firstSince() );
	groups.push_back( m_cache.iovSequence().lastTill() );
	return true;
      }
      return false; 
    }
      
    size_t OraIOVTable::selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime&, 
					      std::vector<cond::Time_t>& groups ){
      // no (easy) way to do it...
      return selectGroups( tag, groups );
    }
      
    size_t OraIOVTable::selectLatestByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup , 
					     std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs){
      if(!m_cache.load( tag )) return 0;
      cond::IOVRange range = m_cache.iovSequence().range( lowerGroup, upperGroup );
      size_t ret = 0;
      for( auto iov : range ){
	iovs.push_back( std::make_tuple( iov.since(), iov.token() ) );
	ret++;
      }
      return ret;
    }

    size_t OraIOVTable::selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
					       const boost::posix_time::ptime&, 
					       std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs){
      // no (easy) way to do it...
      return selectLatestByGroup( tag, lowerGroup, upperGroup, iovs );
    }
    
    size_t OraIOVTable::selectLatest( const std::string& tag, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs){
      // avoid this! copying the entire iov sequence...
      if(!m_cache.load( tag )) return 0;
      size_t ret = 0;
      for( auto iov : m_cache.iovSequence() ){
	iovs.push_back( std::make_tuple( iov.since(), iov.token() ) );
	ret++;
      }
      return ret;      
    }

    bool OraIOVTable::getLastIov( const std::string& tag, cond::Time_t& since, cond::Hash& hash ){
      if(!m_cache.load( tag ) || m_cache.iovSequence().size()==0 ) return false;
      cond::IOVElementProxy last = *(--m_cache.iovSequence().end());
      since = last.since();
      hash = last.token();
      return true;
    }

    bool OraIOVTable::getSize( const std::string& tag, size_t& size ){
      if(!m_cache.load( tag )) return false;
      size = m_cache.iovSequence().size();
      return true;
    }
      
    bool OraIOVTable::getSnapshotSize( const std::string& tag, const boost::posix_time::ptime&, size_t& size ){
      // no (easy) way to do it...
      return getSize( tag,size );
    }
      
    void OraIOVTable::insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, 
				 const boost::posix_time::ptime& ){
      if(!m_cache.load(tag)) throwException("Tag "+tag+" has not been found in the database.",
					    "OraIOVTable::insertOne");
      m_cache.editor().append( since, payloadHash );
    }

    void OraIOVTable::insertMany( const std::string& tag, 
				  const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs ){
      if(!m_cache.load(tag)) throwException("Tag "+tag+" has not been found in the database.",
					    "OraIOVTable::insertOne");
      std::vector<std::pair<cond::Time_t, std::string > > data;
      data.reserve( iovs.size() );
      for( auto v : iovs ){
	data.push_back( std::make_pair( std::get<0>(v), std::get<1>(v) ) );
      }
      m_cache.editor().bulkAppend( data );
    }

    OraIOVSchema::OraIOVSchema( DbSession& session ):
      m_cache( session ),
      m_tagTable( m_cache ),
      m_iovTable( m_cache ),
      m_payloadTable( session ){
    };

    bool OraIOVSchema::exists(){
      return m_cache.session().storage().exists();
    }
    
    bool OraIOVSchema::create(){
      return m_cache.session().createDatabase();
    }
      
    ITagTable& OraIOVSchema::tagTable(){
      return m_tagTable;
    }
      
    IIOVTable& OraIOVSchema::iovTable(){
      return m_iovTable;
    }
      
    IPayloadTable& OraIOVSchema::payloadTable(){
      return m_payloadTable;
    }
      
    ITagMigrationTable& OraIOVSchema::tagMigrationTable(){
      throwException("Tag Migration interface is not available in this implementation.",
		     "OraIOVSchema::tagMigrationTabl");
    }

    OraGTTable::OraGTTable( DbSession& session ):
      m_session( session ){
    }

    bool OraGTTable::select( const std::string& name ){
      cond::TagCollectionRetriever gtRetriever( m_session, "", "" );
      return gtRetriever.existsTagCollection( name+"::All" );
    }
      
    bool OraGTTable::select( const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime){
      bool ret = false;
      if( select( name ) ){
	ret = true;
	validity = cond::time::MAX_VAL;
	snapshotTime = boost::posix_time::ptime();
      }
      return ret;
    }

    bool OraGTTable::select( const std::string& name, cond::Time_t& validity, std::string&, 
			     std::string&, boost::posix_time::ptime& snapshotTime){
      return select( name, validity, snapshotTime );
    }

    void OraGTTable::insert( const std::string&, cond::Time_t, const std::string&, const std::string&, 
			     const boost::posix_time::ptime&, const boost::posix_time::ptime&  ){
      // not supported...
    }
      
    void OraGTTable::update( const std::string&, cond::Time_t, const std::string&, const std::string&, 
			     const boost::posix_time::ptime&, const boost::posix_time::ptime& ){
      // not supported...
    }
     
    OraGTMapTable::OraGTMapTable( DbSession& session ):
      m_session( session ){
    }
      
    bool OraGTMapTable::select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags ){
      return select( gtName, "", "", tags );
    }

    bool OraGTMapTable::select( const std::string& gtName, const std::string& preFix, const std::string& postFix,
				std::vector<std::tuple<std::string,std::string,std::string> >& tags ){
      std::set<cond::TagMetadata> tmp;
      cond::TagCollectionRetriever gtRetriever( m_session, preFix, postFix );
      if(!gtRetriever.selectTagCollection( gtName, tmp ) ) return false;
      if( tmp.size() ) tags.resize( tmp.size() );
      size_t i = 0;
      for( const auto& m : tmp ){
	std::string tagFullName = m.tag+"@["+m.pfn+"]";
	tags[ i ] = std::make_tuple( m.recordname, m.labelname, tagFullName );
	i++;
      }
      return true;
    }
      
    void OraGTMapTable::insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags ){
      // not supported...
    }

    OraGTSchema::OraGTSchema( DbSession& session ):
      m_session( session ),
      m_gtTable( session ),
      m_gtMapTable( session ){
    }
      
    bool OraGTSchema::exists(){
      cond::TagCollectionRetriever gtRetriever( m_session, "", "" );
      return gtRetriever.existsTagDatabase();
    }
    
    IGTTable& OraGTSchema::gtTable(){
      return m_gtTable;
    }

    IGTMapTable& OraGTSchema::gtMapTable(){
      return m_gtMapTable;
    }

  }
}

#include "CondCore/CondDB/interface/GTProxy.h"
#include "SessionImpl.h"

namespace cond {

  namespace persistency {

    std::string fullyQualifiedTag( const std::string& tag, const std::string& connectionString ){
      if(connectionString.empty()) return tag;
      return  tag+"@["+connectionString+"]";
    }

    std::pair<std::string,std::string> parseTag( const std::string& tag ){
      std::string pfn("");
      std::string t(tag);
      size_t pos = tag.rfind("[");
      if( pos != std::string::npos && tag.size() >= pos+2 ){
	if( tag[pos-1]=='@' && tag[tag.size()-1]==']' ) {
	  pfn = tag.substr( pos+1,tag.size()-pos-2 ); 
	  t = tag.substr( 0, pos-1 );
	}
      }
      return std::make_pair( t, pfn );
    }

    // implementation details...
    // only hosting data in this case
    class GTProxyData {
    public: 
      
      GTProxyData():
	name(""),
	preFix(""),
        postFix(""),
	tagList(){
      }
      
      std::string name;
      // will become useless after the transition...
      std::string preFix;
      std::string postFix;
      cond::Time_t validity;
      boost::posix_time::ptime snapshotTime;
      // tag list
      GTProxy::GTContainer tagList;
    };

    GTProxy::Iterator::Iterator():
      m_current(){
    }

    GTProxy::Iterator::Iterator( GTContainer::const_iterator current ):
      m_current( current ){
    }
    
    GTProxy::Iterator::Iterator( const Iterator& rhs ):
      m_current( rhs.m_current ){
    }
    
    GTProxy::Iterator& GTProxy::Iterator::operator=( const Iterator& rhs ){
      if( this != &rhs ){
	m_current = rhs.m_current;
      }
      return *this;
    }

    cond::GTEntry_t GTProxy::Iterator::operator*() {
      return cond::GTEntry_t( *m_current );
    }
    
    GTProxy::Iterator& GTProxy::Iterator::operator++(){
      m_current++;
      return *this;
    }
    
    GTProxy::Iterator GTProxy::Iterator::operator++(int){
      Iterator tmp( *this );
      operator++();
      return tmp;
    }
    
    bool GTProxy::Iterator::operator==( const Iterator& rhs ) const {
      if( m_current != rhs.m_current ) return false;
      return true;
    }
    
    bool GTProxy::Iterator::operator!=( const Iterator& rhs ) const {
      return !operator==( rhs );
    }
    
    GTProxy::GTProxy():
      m_data(),
      m_session(){
    }
    
    GTProxy::GTProxy( const std::shared_ptr<SessionImpl>& session ):
      m_data( new GTProxyData ),
      m_session( session ){
    }
    
    GTProxy::GTProxy( const GTProxy& rhs ):
      m_data( rhs.m_data ),
      m_session( rhs.m_session ){
    }
    
    GTProxy& GTProxy::operator=( const GTProxy& rhs ){
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }
    
    /** this will be the final function 
    void GTProxy::load( const std::string& gtName ){
      // clear
      reset();
      
      checkSession( "GTProxy::load" );
      
      if(!m_session->gtSchema().gtTable().select( gtName, m_data->validity, m_data->snapshotTime ) ){
	throwException( "Global Tag \""+gtName+"\" has not been found in the database.","GTProxy::load");
      }
      m_data->name = gtName;
      
      m_session->gtSchema().gtMapTable().select( m_data->name, m_data->tagList );
      
    }
    **/

      // overloading for pre- and post-fix. Used in the ORA implementation
    void GTProxy::load( const std::string& gtName, const std::string& pref, const std::string& postf ){
      // clear
      reset();
      
      checkTransaction( "GTProxy::load" );
      
      if(!m_session->gtSchema().gtTable().select( gtName, m_data->validity, m_data->snapshotTime ) ){
	throwException( "Global Tag \""+gtName+"\" has not been found in the database.","GTProxy::load");
      }
      m_data->name = gtName;
      m_data->preFix = pref;
      m_data->postFix = postf;

      m_session->gtSchema().gtMapTable().select( m_data->name, pref, postf, m_data->tagList );

    }

    
    void GTProxy::reload(){
      load( m_data->name );
    }
    
    void GTProxy::reset(){
      if( m_data.get() ){
	m_data->tagList.clear();
      }
    }
    
    std::string GTProxy::name() const {
      return m_data.get() ? m_data->name : std::string("");
    }
    
    cond::Time_t GTProxy::validity() const {
      return m_data.get() ? m_data->validity : cond::time::MIN_VAL;
    }
    
    boost::posix_time::ptime GTProxy::snapshotTime() const {
      return m_data.get() ? m_data->snapshotTime : boost::posix_time::ptime();
    }
    
    void GTProxy::checkTransaction( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
    GTProxy::Iterator GTProxy::begin() const {
      if( m_data.get() ){
	return Iterator( m_data->tagList.begin() );
      } 
      return Iterator();
    }
    
    GTProxy::Iterator GTProxy::end() const {
      if( m_data.get() ){
	return Iterator( m_data->tagList.end() );
      } 
      return Iterator();
    }
    
    int GTProxy::size() const {
      return m_data.get()? m_data->tagList.size() : 0;
    }

  }
}

#include "CondCore/CondDB/interface/GTEditor.h"
#include "SessionImpl.h"
//

namespace cond {

  namespace persistency {

    // GT data...
    class GTEditorData {
    public:
      explicit GTEditorData():
	name( "" ),
	description(""),
	release(""),
	snapshotTime(),
	tagListBuffer(){
      }
      // GT data
      std::string name;
      cond::Time_t validity = cond::time::MAX_VAL;
      std::string description;
      std::string release;
      boost::posix_time::ptime snapshotTime;
      bool change = false;
      bool exists = false;
      // buffer for the GT tag map
      std::vector<std::tuple<std::string,std::string,std::string> > tagListBuffer;
    };

    GTEditor::GTEditor( const std::shared_ptr<SessionImpl>& session ):
      m_data( new GTEditorData ),
      m_session( session ){
    }

    GTEditor::GTEditor( const std::shared_ptr<SessionImpl>& session, 
			const std::string& gtName ):
      m_data( new GTEditorData ),
      m_session( session ){
      m_data->name = gtName;
      m_data->change = true;
    }
    
    GTEditor::GTEditor( const GTEditor& rhs ):
      m_data( rhs.m_data ){
    }
    
    GTEditor& GTEditor::operator=( const GTEditor& rhs ){
      m_data = rhs.m_data;
      return *this;
    }
    
    void GTEditor::load( const std::string& gtName ){
      checkTransaction( "GTEditor::load" );
      
      // loads the current header data in memory
      if( !m_session->gtSchema().gtTable().select( gtName, m_data->validity, m_data->description, m_data->release, m_data->snapshotTime ) ){
	throwException( "Global Tag \""+gtName+"\" has not been found in the database.","GTEditor::load");
      }
      m_data->name = gtName;
      m_data->exists = true;
      m_data->change = false;
    }
    
    std::string GTEditor::name() const {
      return m_data.get()? m_data->name : "" ;
    }
    
    cond::Time_t GTEditor::validity() const {
      return m_data.get()? m_data->validity : cond::time::MIN_VAL;
    }
    
    void GTEditor::setValidity( cond::Time_t validity ){
      if( m_data.get() ) {
	m_data->validity = validity;
	m_data->change = true;
      }
    }
    
    std::string GTEditor::description() const {
      return m_data.get()? m_data->description : "";
    }
    
    void GTEditor::setDescription( const std::string& description ){
      if( m_data.get() ) {
	m_data->description = description;
	m_data->change = true;
      }
    }
    
    std::string GTEditor::release() const {
      return m_data.get()? m_data->release : "";
    }
    
    void GTEditor::setRelease( const std::string& release ){
      if( m_data.get() ) {
	m_data->release = release;
	m_data->change = true;
      }
    }
    
    boost::posix_time::ptime GTEditor::snapshotTime() const {
      return m_data.get()? m_data->snapshotTime : boost::posix_time::ptime(); 
    }
    
    void GTEditor::setSnapshotTime( const boost::posix_time::ptime& snapshotTime ){
      if( m_data.get() ) {
	m_data->snapshotTime = snapshotTime;
	m_data->change = true;
      }
    }
    
    void GTEditor::insert( const std::string& recordName, const std::string& tagName, bool checkType ){
      insert( recordName, "-", tagName, checkType );
    }
    
    void GTEditor::insert( const std::string& recordName, const std::string& recordLabel, const std::string& tagName, bool ){
      if( m_data.get() ){
	// here the type check could be added                                                                                              
        
	std::string rlabel = recordLabel;
	if( rlabel.empty() ){
	  rlabel = "-";
	}
	m_data->tagListBuffer.push_back( std::tie( recordName, rlabel, tagName ) );
      } 
    }
    
    bool GTEditor::flush( const boost::posix_time::ptime& operationTime ){
      bool ret = false;
      checkTransaction( "GTEditor::flush" );
      if( m_data->change ){
	if( m_data->description.empty() ) throwException( "A non-empty Description string is mandatory.","GTEditor::flush" );
	if( m_data->release.empty() ) throwException( "A non-empty Release string is mandatory.","GTEditor::flush" );
	if( !m_data->exists ){
	  m_session->gtSchema().gtTable().insert( m_data->name, m_data->validity, m_data->description, 
						  m_data->release, m_data->snapshotTime, operationTime );
	  ret = true;
	  m_data->exists = true;
	} else {
	  m_session->gtSchema().gtTable().update( m_data->name,  m_data->validity, m_data->description, 
						  m_data->release,m_data->snapshotTime, operationTime );
	  ret = true;
	}
	m_data->change = false;  
      }
      if( m_data->tagListBuffer.size() ) {
	
	// insert the new iovs
	m_session->gtSchema().gtMapTable().insert( m_data->name, m_data->tagListBuffer );
	m_data->tagListBuffer.clear();
	ret = true;
      }
      return ret;
    }
    
    bool GTEditor::flush(){
      return flush( boost::posix_time::microsec_clock::universal_time() );
    }
    
    
    void GTEditor::checkTransaction( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
  }
}

  

#include "CondCore/CondDB/interface/IOVEditor.h"
#include "SessionImpl.h"
#include "IOVSchema.h"
//

namespace cond {

  namespace persistency {

    // implementation details. holds only data.
    class IOVEditorData {
    public:
      explicit IOVEditorData():
	tag( "" ),
	timeType( cond::invalid ),
	payloadType(""),
	synchronizationType( cond::OFFLINE ),
	description(""),
	iovBuffer(){
      }
      std::string tag;
      cond::TimeType timeType;
      std::string payloadType;
      cond::SynchronizationType synchronizationType; 
      std::string description;
      cond::Time_t endOfValidity = cond::time::MAX;
      cond::Time_t lastValidatedTime = cond::time::MIN; 
      bool change = false;
      bool exists = false;
      // buffer for the iov sequence
      std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> > iovBuffer;
    };

    IOVEditor::IOVEditor():
      m_data(),
      m_session(){
    }

    IOVEditor::IOVEditor( const std::shared_ptr<SessionImpl>& session ):
      m_data( new IOVEditorData ),
      m_session( session ){
    }

    IOVEditor::IOVEditor( const std::shared_ptr<SessionImpl>& session, 
			  const std::string& tag, 
			  cond::TimeType timeType, 
			  const std::string& payloadObjectType,
			  cond::SynchronizationType synchronizationType  ):
      m_data( new IOVEditorData ),
      m_session( session ){
      m_data->tag = tag;
      m_data->timeType = timeType;
      m_data->payloadType = payloadObjectType;
      m_data->synchronizationType = synchronizationType;
      m_data->change = true;
    }

    IOVEditor::IOVEditor( const IOVEditor& rhs ):
      m_data( rhs.m_data ),
      m_session( rhs.m_session ){
    }
    
    IOVEditor& IOVEditor::operator=( const IOVEditor& rhs ){
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }
    
    void IOVEditor::load( const std::string& tag ){
      checkSession( "IOVEditor::load" );
      // loads the current header data in memory
      if( !TAG::select( tag, m_data->timeType, m_data->payloadType, m_data->endOfValidity, m_data->description, m_data->lastValidatedTime, *m_session ) ){
	cond::throwException( "Tag \""+tag+"\" has not been found in the database.","IOVEditor::load");
      }
      m_data->tag = tag;
      m_data->exists = true;
      m_data->change = false;
    }
    
    std::string IOVEditor::tag() const {
      return m_data.get()? m_data->tag : "" ;
    }
    
    
    cond::TimeType IOVEditor::timeType() const {
      return m_data.get() ? m_data->timeType : cond::invalid;
    }
    
    std::string IOVEditor::payloadType() const {
      return m_data.get() ? m_data->payloadType :  "";
    }
    
    cond::SynchronizationType IOVEditor::synchronizationType() const {
      return m_data.get()? m_data->synchronizationType : cond::SYNCHRONIZATION_UNKNOWN ; 
    }
    
    cond::Time_t IOVEditor::endOfValidity() const {
      return m_data.get() ? m_data->endOfValidity : cond::time::MIN;
    }
    
    void IOVEditor::setEndOfValidity( cond::Time_t time ){
      if( m_data.get() ) {
	m_data->endOfValidity = time;
	m_data->change = true;
      }
    }
    
    std::string IOVEditor::description() const {
      return m_data.get() ? m_data->description :  "";
    }
    
    void IOVEditor::setDescription( const std::string& description ){
      if( m_data.get() ) {
	m_data->description = description;
	m_data->change = true;
      }
    }
    
    cond::Time_t IOVEditor::lastValidatedTime() const {
      return m_data.get() ? m_data->lastValidatedTime : cond::time::MIN;
    }
    
    void IOVEditor::setLastValidatedTime(cond::Time_t time ){
      if( m_data.get() ) {
	m_data->lastValidatedTime = time;
	m_data->change = true;
      }
    }
    
    void IOVEditor::insert( cond::Time_t since, const cond::Hash& payloadHash, bool checkType ){
      boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
      insert( since, payloadHash, now, checkType ); 
    }
    
    void IOVEditor::insert( cond::Time_t since, const cond::Hash& payloadHash, const boost::posix_time::ptime& insertionTime, bool ){
      if( m_data.get() ){
	// here the type check could be added
	m_data->iovBuffer.push_back( std::tie( since, payloadHash, insertionTime ) );
      }
    }
    
    bool IOVEditor::flush( const boost::posix_time::ptime& operationTime ){
      bool ret = false;
      checkSession( "IOVEditor::flush" );
      if( m_data->change ){
	if( m_data->description.empty() ) throwException( "A non-empty description string is mandatory.","IOVEditor::flush" );
	if( !m_data->exists ){
	  TAG::insert( m_data->tag, m_data->timeType, m_data->payloadType, m_data->synchronizationType, m_data->endOfValidity, 
		       m_data->description, m_data->lastValidatedTime, operationTime, *m_session );
	  m_data->exists = true;
	  ret = true;
	} else {
	  TAG::update( m_data->tag, m_data->endOfValidity, m_data->description, m_data->lastValidatedTime, operationTime, *m_session );   
	  ret = true;
	}
	m_data->change = false;
      }
      if( m_data->iovBuffer.size() ) {
	
	// insert the new iovs
	IOV::insertMany( m_data->tag, m_data->iovBuffer, *m_session );
	m_data->iovBuffer.clear();
	ret = true;
      }
      return ret;
    }
    
    bool IOVEditor::flush(){
      return flush( boost::posix_time::microsec_clock::universal_time() );
    }
    
    void IOVEditor::checkSession( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
    }
    
  }
}

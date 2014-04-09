#include "CondCore/CondDB/interface/IOVEditor.h"
#include "SessionImpl.h"
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
      cond::Time_t endOfValidity = cond::time::MAX_VAL;
      cond::Time_t lastValidatedTime = cond::time::MIN_VAL; 
      bool change = false;
      bool exists = false;
      // buffer for the iov sequence
      std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> > iovBuffer;
      bool validationMode = false;
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
      checkTransaction( "IOVEditor::load" );
      // loads the current header data in memory
      if( !m_session->iovSchema().tagTable().select( tag, m_data->timeType, m_data->payloadType, m_data->synchronizationType, 
						     m_data->endOfValidity, m_data->description, m_data->lastValidatedTime ) ){
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
      return m_data.get() ? m_data->endOfValidity : cond::time::MIN_VAL;
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
      return m_data.get() ? m_data->lastValidatedTime : cond::time::MIN_VAL;
    }
    
    void IOVEditor::setLastValidatedTime(cond::Time_t time ){
      if( m_data.get() ) {
	m_data->lastValidatedTime = time;
	m_data->change = true;
      }
    }

    void IOVEditor::setValidationMode(){
      if( m_data.get() ) m_data->validationMode = true;
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
      checkTransaction( "IOVEditor::flush" );
      if( m_data->change ){
	if( m_data->description.empty() ) throwException( "A non-empty description string is mandatory.","IOVEditor::flush" );
	if( m_data->validationMode ) m_session->iovSchema().tagTable().setValidationMode();
	if( !m_data->exists ){
	  m_session->iovSchema().tagTable().insert( m_data->tag, m_data->timeType, m_data->payloadType, 
						    m_data->synchronizationType, m_data->endOfValidity, 
						    m_data->description, m_data->lastValidatedTime, operationTime );
	  m_data->exists = true;
	  ret = true;
	} else {
	  m_session->iovSchema().tagTable().update( m_data->tag, m_data->endOfValidity, m_data->description, 
						    m_data->lastValidatedTime, operationTime );   
	  ret = true;
	}
	m_data->change = false;
      }
      if( m_data->iovBuffer.size() ) {
	
	// insert the new iovs
	m_session->iovSchema().iovTable().insertMany( m_data->tag, m_data->iovBuffer );
	m_data->iovBuffer.clear();
	ret = true;
      }
      return ret;
    }
    
    bool IOVEditor::flush(){
      return flush( boost::posix_time::microsec_clock::universal_time() );
    }
    
    void IOVEditor::checkTransaction( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
  }
}

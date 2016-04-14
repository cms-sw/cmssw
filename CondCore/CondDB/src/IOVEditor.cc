#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/Utils.h"
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
	synchronizationType( cond::SYNCH_ANY ),
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
      boost::posix_time::ptime creationTime;
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
			  cond::SynchronizationType synchronizationType,
			  const boost::posix_time::ptime& creationTime ):
      m_data( new IOVEditorData ),
      m_session( session ){
      m_data->tag = tag;
      m_data->timeType = timeType;
      m_data->payloadType = payloadObjectType;
      m_data->synchronizationType = synchronizationType;
      m_data->creationTime = creationTime;
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
      return m_data.get()? m_data->synchronizationType : cond::SYNCH_ANY ; 
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

    bool iovSorter( const std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime>& f, const std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime>& s ){
      return std::get<0>(f) < std::get<0>(s);
    }

    bool IOVEditor::flush( const std::string& logText, const boost::posix_time::ptime& operationTime ){
      bool ret = false;
      checkTransaction( "IOVEditor::flush" );
      if( m_data->change ){
	if( m_data->description.empty() ) throwException( "A non-empty description string is mandatory.","IOVEditor::flush" );
	if( m_data->validationMode ) m_session->iovSchema().tagTable().setValidationMode();
	if( !m_data->exists ){
	  // set the creation time ( only available in the migration from v1...)
	  if( m_data->creationTime.is_not_a_date_time() )  m_data->creationTime = operationTime;
	  m_session->iovSchema().tagTable().insert( m_data->tag, m_data->timeType, m_data->payloadType, 
						    m_data->synchronizationType, m_data->endOfValidity, 
						    m_data->description, m_data->lastValidatedTime, m_data->creationTime );
          if( m_session->iovSchema().tagLogTable().exists() )
	    m_session->iovSchema().tagLogTable().insert(  m_data->tag,  m_data->creationTime, cond::getUserName(),cond::getHostName(), cond::getCommand(),
							  std::string("New tag created."),std::string("-") );
	  m_data->exists = true;
	  ret = true;
	} else {
	  m_session->iovSchema().tagTable().update( m_data->tag, m_data->endOfValidity, m_data->description, 
						    m_data->lastValidatedTime, operationTime );   
          if( m_session->iovSchema().tagLogTable().exists() )
	    m_session->iovSchema().tagLogTable().insert(  m_data->tag,  operationTime, cond::getUserName(),cond::getHostName(), cond::getCommand(), 
							  std::string("Tag header updated."),std::string("-") );
	  ret = true;
	}
	m_data->change = false;
      }
      if( m_data->iovBuffer.size() ) {
	std::sort(m_data->iovBuffer.begin(),m_data->iovBuffer.end(),iovSorter);
	cond::Time_t l = std::get<0>(m_data->iovBuffer.front());
	if( m_data->synchronizationType != cond::SYNCH_ANY && m_data->synchronizationType != cond::SYNCH_VALIDATION ){
	  // retrieve the last since
	  cond::Time_t last = 0;
	  cond::Hash h;
	  m_session->iovSchema().iovTable().getLastIov( m_data->tag, last, h );
	  // check if the min iov is greater then the last since
	  if( l <= last ){
	    std::stringstream msg;
	    msg << "Can't insert iov since "<<l<<" on the tag "<< m_data->tag<<": last since is "<<last<<
	      " and synchronization is \""<<cond::synchronizationTypeNames(  m_data->synchronizationType )<<"\"";
	    throwException( msg.str(),"IOVEditor::flush");
	  }
	}
	// set the insertion time ( only for the migration from v1 will be available... )
        for( auto& iov : m_data->iovBuffer ){
	  boost::posix_time::ptime& insertionTime = std::get<2>(iov);
	  if( insertionTime.is_not_a_date_time() ) insertionTime = operationTime;
	}
	// insert the new iovs
	m_session->iovSchema().iovTable().insertMany( m_data->tag, m_data->iovBuffer );
	std::stringstream msg;
        msg << m_data->iovBuffer.size() << " iov(s) inserted.";
	if( m_session->iovSchema().tagLogTable().exists() ){
	  std::string lt = logText;
	  if( lt.empty() ) lt = "-";
	  m_session->iovSchema().tagLogTable().insert(  m_data->tag,  operationTime, cond::getUserName(), cond::getHostName(), cond::getCommand(), 
							msg.str(),lt );
	}
	m_data->iovBuffer.clear();
	ret = true;
      }
      return ret;
    }
    
    bool IOVEditor::flush( const std::string& logText ){
      return flush( logText, boost::posix_time::microsec_clock::universal_time() );
    }

    bool IOVEditor::flush( const boost::posix_time::ptime& operationTime ){
      return flush( std::string("-"), operationTime );
    }

    bool IOVEditor::flush(){
      return flush( std::string("-"), boost::posix_time::microsec_clock::universal_time() );
    }

    void IOVEditor::checkTransaction( const std::string& ctx ){
      if( !m_session.get() ) throwException("The session is not active.",ctx );
      if( !m_session->isTransactionActive( false ) ) throwException("The transaction is not active.",ctx );
    }
    
  }
}

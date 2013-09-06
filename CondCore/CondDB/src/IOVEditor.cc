#include "CondCore/CondDB/interface/IOVEditor.h"
#include "SessionImpl.h"
#include "IOVSchema.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/shared_ptr.hpp>

namespace new_impl {

  // implementation details. holds only data.
  class IOVEditorData {
  public:
    explicit IOVEditorData():
      tag( "" ),
      timeType( conddb::time::INVALID ),
      payloadType(""),
      synchronizationType( conddb::OFFLINE ),
      description(""),
      iovBuffer(){
    }
    std::string tag;
    conddb::TimeType timeType;
    std::string payloadType;
    conddb::SynchronizationType synchronizationType; 
    std::string description;
    conddb::Time_t endOfValidity = conddb::time::MAX;
    conddb::Time_t lastValidatedTime = conddb::time::MIN; 
    bool change = false;
    bool exists = false;
    // buffer for the iov sequence
    std::vector<std::tuple<conddb::Time_t,conddb::Hash,boost::posix_time::ptime> > iovBuffer;
  };

IOVEditor::IOVEditor():
  m_data(),
  m_session(){
}

IOVEditor::IOVEditor( const boost::shared_ptr<conddb::SessionImpl>& session ):
  m_data( new IOVEditorData ),
  m_session( session ){
}

IOVEditor::IOVEditor( const boost::shared_ptr<conddb::SessionImpl>& session, 
		      const std::string& tag, 
		      conddb::TimeType timeType, 
		      const std::string& payloadObjectType,
		      conddb::SynchronizationType synchronizationType  ):
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
  if( !conddb::TAG::select( tag, m_data->timeType, m_data->payloadType, m_data->endOfValidity, m_data->description, m_data->lastValidatedTime, *m_session ) ){
    conddb::throwException( "Tag \""+tag+"\" has not been found in the database.","IOVEditor::load");
  }
  m_data->tag = tag;
  m_data->exists = true;
  m_data->change = false;
}

std::string IOVEditor::tag() const {
  return m_data.get()? m_data->tag : "" ;
}
    

conddb::TimeType IOVEditor::timeType() const {
  return m_data.get() ? m_data->timeType : conddb::time::INVALID;
}

std::string IOVEditor::payloadType() const {
  return m_data.get() ? m_data->payloadType :  "";
}

conddb::SynchronizationType IOVEditor::synchronizationType() const {
  return m_data.get()? m_data->synchronizationType : conddb::SYNCHRONIZATION_UNKNOWN ; 
}

conddb::Time_t IOVEditor::endOfValidity() const {
  return m_data.get() ? m_data->endOfValidity : conddb::time::MIN;
}

void IOVEditor::setEndOfValidity( conddb::Time_t time ){
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

conddb::Time_t IOVEditor::lastValidatedTime() const {
  return m_data.get() ? m_data->lastValidatedTime : conddb::time::MIN;
}

void IOVEditor::setLastValidatedTime(conddb::Time_t time ){
  if( m_data.get() ) {
    m_data->lastValidatedTime = time;
    m_data->change = true;
  }
}

void IOVEditor::insert( conddb::Time_t since, const conddb::Hash& payloadHash, bool checkType ){
  boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
  insert( since, payloadHash, now, checkType ); 
}

void IOVEditor::insert( conddb::Time_t since, const conddb::Hash& payloadHash, const boost::posix_time::ptime& insertionTime, bool ){
  if( m_data.get() ){
    // here the type check could be added                                                                                                                                             
    m_data->iovBuffer.push_back( std::tie( since, payloadHash, insertionTime ) );
  }
}

bool IOVEditor::flush( const boost::posix_time::ptime& operationTime ){
  bool ret = false;
  checkSession( "IOVEditor::flush" );
  if( m_data->change ){
    if( m_data->description.empty() ) conddb::throwException( "A non-empty description string is mandatory.","IOVEditor::flush" );
    if( !m_data->exists ){
      conddb::TAG::insert( m_data->tag, m_data->timeType, m_data->payloadType, m_data->synchronizationType, m_data->endOfValidity, 
		   m_data->description, m_data->lastValidatedTime, operationTime, *m_session );
      m_data->exists = true;
      ret = true;
    } else {
      conddb::TAG::update( m_data->tag, m_data->endOfValidity, m_data->description, m_data->lastValidatedTime, operationTime, *m_session );   
      ret = true;
    }
    m_data->change = false;
  }
  if( m_data->iovBuffer.size() ) {

    // insert the new iovs
    conddb::IOV::insertMany( m_data->tag, m_data->iovBuffer, *m_session );
    m_data->iovBuffer.clear();
    ret = true;
  }
  return ret;
}

bool IOVEditor::flush(){
  return flush( boost::posix_time::microsec_clock::universal_time() );
}

void IOVEditor::checkSession( const std::string& ctx ){
  if( !m_session.get() ) conddb::throwException("The session is not active.",ctx );
}

}

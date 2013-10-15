#include "CondCore/CondDB/interface/PayloadProxy.h"

namespace cond {

  namespace db {

    BasePayloadProxy::BasePayloadProxy( cond::db::Session& session ) :
      m_iovProxy(),m_session(session) {
    }

    BasePayloadProxy::BasePayloadProxy(cond::db::Session& session,
				       const std::string& tag ) :
      m_iovProxy(),m_session(session) {
      m_session.transaction().start(true);
      m_iovProxy = m_session.readIov( tag );
      m_session.transaction().commit();
    }
    
    BasePayloadProxy::~BasePayloadProxy(){}
    
    void BasePayloadProxy::loadTag( const std::string& tag ){
      m_session.transaction().start(true);
      m_iovProxy = m_session.readIov( tag );
      m_session.transaction().commit();
      invalidateCache();
    }
    
    void BasePayloadProxy::reload(){
      m_session.transaction().start(true);
      m_iovProxy.reload();
      m_session.transaction().commit();
      invalidateCache();    
    }
    
    ValidityInterval BasePayloadProxy::setIntervalFor(cond::Time_t time) {
      if( !m_currentIov.isValidFor( time ) ){
	m_session.transaction().start(true);
	auto it = m_iovProxy.find( time );
	if( it == m_iovProxy.end() ) throwException( "No valid iov found for time "+boost::lexical_cast<std::string>( time ),
						     "BasePayloadProxy::setIntervalFor" );
	m_currentIov = *it;
	loadPayload();
	m_session.transaction().commit();
      }
      return ValidityInterval( m_currentIov.since, m_currentIov.till );
    }
    
    bool BasePayloadProxy::isValid() const {
      return m_currentIov.isValid();
    }

  }
}

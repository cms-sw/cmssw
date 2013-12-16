#include "CondCore/CondDB/interface/PayloadProxy.h"

namespace cond {

  namespace persistency {

    BasePayloadProxy::BasePayloadProxy() :
      m_iovProxy(),m_session() {
    }

    BasePayloadProxy::~BasePayloadProxy(){}

    void BasePayloadProxy::setUp( Session dbSession ){
      m_session = dbSession;
      //m_iovProxy = m_session.iovProxy();
      invalidateCache();    
    }
    
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
    
    ValidityInterval BasePayloadProxy::setIntervalFor(cond::Time_t time, bool load) {
      if( !m_currentIov.isValidFor( time ) ){
	m_session.transaction().start(true);
	auto it = m_iovProxy.find( time );
	if( it == m_iovProxy.end() ) {
	  throwException( "No valid iov found in tag "+m_iovProxy.tag()+" for time "+boost::lexical_cast<std::string>( time ),
	  		  "BasePayloadProxy::setIntervalFor" );
	}
	m_currentIov = *it;
	if(load) loadPayload();
	m_session.transaction().commit();
      }
      return ValidityInterval( m_currentIov.since, m_currentIov.till );
    }
    
    bool BasePayloadProxy::isValid() const {
      return m_currentIov.isValid();
    }

    IOVProxy BasePayloadProxy::iov() {
      return m_iovProxy;
    }

  }
}

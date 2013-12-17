#include "CondCore/CondDB/interface/PayloadProxy.h"

namespace cond {

  namespace persistency {

    BasePayloadProxy::BasePayloadProxy() :
      m_iovProxy(),m_session() {
    }

    BasePayloadProxy::~BasePayloadProxy(){}

    void BasePayloadProxy::setUp( Session dbSession ){
      m_session = dbSession;
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
	  std::stringstream msg;
	  msg << "No valid iov found";
	  switch ( m_iovProxy.timeType() ){
	  case cond::time::RUNNUMBER:
	    msg <<" for run "<<time;
	    break; 
          case cond::time::TIMESTAMP:
	    msg <<" for time "<<time;
	    break; 
	  case cond::time::LUMIID:
	    msg <<" for run "<<cond::time::unpack(time).first<<", lumisection "<<cond::time::unpack(time).second;
	    break; 
          case cond::time::HASH:
	    msg <<" for hash "<<time;
	    break; 
	  case cond::time::USERID:
	    msg <<"for userid "<<time;
	    break; 
	  case cond::time::INVALID: 
	    msg <<", invalid timetype";
	    break;
	  }
	  msg <<" in tag "<<m_iovProxy.tag();
	  throwException( msg.str(),
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

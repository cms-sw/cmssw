#include "CondCore/IOVService/interface/PayloadProxy.h"

#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "DataSvc/RefException.h"


namespace cond {


  BasePayloadProxy::BasePayloadProxy(cond::Connection& conn,
				     const std::string & token, bool errorPolicy) :
    m_doThrow(errorPolicy), m_iov(conn,token,true,false) {
    
  }


  BasePayloadProxy::~BasePayloadProxy(){}

  void BasePayloadProxy::loadFor(cond::Time_t time) {
    m_element = *m_iov.find(time);
    make();
  }

  void  BasePayloadProxy::make() {
    bool ok = false;
    if ( isValid()) {
      cond::PoolTransaction & db = *m_element.db();
      db.start(true);
      try {
	ok = load(&db.poolDataSvc(),m_element.token());
      }	catch( const pool::Exception& e) {
	if (m_doThrow) throw cond::Exception(std::string("Condition Payload loader: ")+ e.what());
	ok = false;
      }
      db.commit();
    }

    if (!ok) {
      m_element.set(cond::invalidTime,cond::invalidTime,"");
      if (m_doThrow)
	throw cond::Exception("Condition Payload loader: invalid data");
    }
  }


  cond::ValidityInterval BasePayloadProxy::setIntervalFor(cond::Time_t time) {
    //FIXME: shall handle truncation...
    if ( (!(time<m_element.till())) || time<m_element.since() )
      m_element = *m_iov.find(time);
    return cond::ValidityInterval(m_element.since(),m_element.till());
  }
    
  bool BasePayloadProxy::isValid() const {
    return m_element.since()!=cond::invalidTime && m_element.till()!=cond::invalidTime
      &&  !m_element.token().empty();
  }


  bool  BasePayloadProxy::refresh() {
    return m_iov.refresh();
  }





}

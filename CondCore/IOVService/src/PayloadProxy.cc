#include "CondCore/IOVService/interface/PayloadProxy.h"

#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "DataSvc/RefException.h"


namespace cond {

  BasePayloadProxy::Stats BasePayloadProxy::stats = {0,0,0};


  BasePayloadProxy::BasePayloadProxy(cond::DbSession& session,
                                     const std::string & token,
                                     bool errorPolicy) :
    m_doThrow(errorPolicy), m_iov(session,token,true,false) {
    ++stats.nProxy;
  }


  BasePayloadProxy::~BasePayloadProxy(){}

  cond::ValidityInterval BasePayloadProxy::loadFor(cond::Time_t time) {
    m_element = *m_iov.find(time);
    make();
    return cond::ValidityInterval(m_element.since(),m_element.till());
  }

  cond::ValidityInterval BasePayloadProxy::loadFor(size_t n) {
    m_element.set(m_iov.iov(),n);
    make();
    return cond::ValidityInterval(m_element.since(),m_element.till());
  }


  void  BasePayloadProxy::make() {
    ++stats.nMake;
    bool ok = false;
    if ( isValid()) {
      // check if (afterall) the payload is still the same...
      if (m_element.token()==token()) return;
      cond::DbTransaction& trans = m_element.db().transaction();
      trans.start(true);
      try {
        ok = load(&m_element.db().poolCache(),m_element.token());
	if (ok) m_token = m_element.token();
      }	catch( const pool::Exception& e) {
        if (m_doThrow) throw cond::Exception(std::string("Condition Payload loader: ")+ e.what());
        ok = false;
      }
      trans.commit();
    }

    if (!ok) {
      m_element.set(cond::invalidTime,cond::invalidTime,"");
      if (m_doThrow)
        throw cond::Exception("Condition Payload loader: invalid data");
    }
    if (ok)  ++stats.nLoad;
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
    bool anew = m_iov.refresh();
    if (anew)  m_element = IOVElementProxy();
    return anew;
  }





}

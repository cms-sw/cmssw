#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/ClassInfoLoader.h"

#include "IOV.h"



namespace cond {

  namespace impl {
    struct IOVImpl {
      IOVImpl(cond::PoolTransaction& db,
	      const std::string & token) :
	pooldb(db){
	db.start(true);
	iov = cond::TypedRef<cond::IOV>(db,token);
	if (iov->iov.empty()) return;
	// load dict
	std::string ptok = iov->iov.front().second;
	db.commit();   
	cond::reflexTypeByToken(ptok);
	db.start(true);
	iov = cond::TypedRef<cond::IOV>(db,token);
      }
      ~IOVImpl(){
	pooldb.commit();
      }
      cond::PoolTransaction & pooldb;
      cond::TypedRef<cond::IOV> iov;
    };

  }


  IOVProxy::IterHelp::IterHelp(impl::IOVImpl & impl) :
    iov(*impl.iov), elem(&impl.pooldb){}
  

  void IOVElement::set(IOV const & v, int i) {
    m_since = (i==0) ? v.firstsince : v.iov[i-1].first+1;
    m_till  = v.iov[i].first;
    m_token = v.iov[i].second;
  }



  IOVProxy::IOVProxy(){}
 
  IOVProxy::~IOVProxy() {}

  IOVProxy::IOVProxy(cond::PoolTransaction& db,
		     const std::string & token) :
    m_iov(new impl::IOVImpl(db,token)){}


  int IOVProxy::size() const {
    return iov().iov.size();
  }

  IOV const & IOVProxy::iov() const {
    return *(*m_iov).iov;
  }

  TimeType IOVProxy::timetype() const {
    return (TimeType)(iov().timetype);     
  }

}

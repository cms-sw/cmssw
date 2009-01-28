#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/ClassInfoLoader.h"

#include "CondFormats/Common/interface/IOVSequence.h"




namespace cond {

  namespace impl {
    struct IOVImpl {
      IOVImpl(cond::PoolTransaction& db,
	      const std::string & token,
	      bool nolib) :
	pooldb(db){
	db.start(true);
	iov = cond::TypedRef<cond::IOVSequence>(db,token);
	if (iov->iovs().empty() || nolib) return;
	// load dict (change: use IOV metadata....)
	std::string ptok = iov->iovs().front().wrapperToken();
	db.commit();   
	cond::reflexTypeByToken(ptok);
	db.start(true);
	iov = cond::TypedRef<cond::IOVSequence>(db,token);
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
  

  void IOVElementProxy::set(IOVSequence const & v, int i) {
    m_since =  v.iovs()[i].sinceTime();
    m_till  =  (i+1==v.iovs().size()) ? v.lastTill() : v.iov[i+1].sinceTime()-1;
    m_token = v.iovs()[i].wrapperToken();
  }



  IOVProxy::IOVProxy() : m_low(0), m_high(0){}
 
  IOVProxy::~IOVProxy() {}

  IOVProxy::IOVProxy(cond::PoolTransaction& db,
		     const std::string & token, bool nolib) :
    m_iov(new impl::IOVImpl(db,token,nolib)), m_low(0), m_high(size()){}


  void IOVProxy::setRange(cond::Time_t since, cond::Time_t  till) const {
    m_low=iov().find(since)-iov().iov.begin();
    m_high=iov().find(till)-iov().iov.begin();
    m_high=std::min(m_high+1,size());
  }

  void IOVProxy::head(int n) const {
    m_high = std::min(m_low+n,m_high);
  }

  void  IOVProxy::tail(int n) const {
    m_low = std::max(m_high-n,m_low);
  }



  int IOVProxy::size() const {
    return iov().iovs().size();
  }

  IOV const & IOVProxy::iov() const {
    return *(*m_iov).iov;
  }

  TimeType IOVProxy::timetype() const {
    return iovs().timeType();     
  }

}

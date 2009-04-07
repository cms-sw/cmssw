#include "CondCore/IOVService/interface/IOVProxy.h"
#include "DataSvc/Ref.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/ClassInfoLoader.h"

#include "CondFormats/Common/interface/IOVSequence.h"




namespace cond {

  namespace impl {
    struct IOVImpl {
      IOVImpl(cond::Connection& conn,
	      const std::string & token,
	      bool nolib,
	      bool keepOpen) :
	connection(conn){
	pooldb().start(true);
	pool::Ref<cond::IOVSequence> temp(&(pooldb().poolDataSvc()),token);
	iov.copyShallow(temp);
	pooldb().commit();   
	if (!iov->iovs().empty() && !nolib) {
	  // load dict (change: use IOV metadata....)
	  std::string ptok = iov->iovs().front().wrapperToken();
	  cond::reflexTypeByToken(ptok);
	}
	if (keepOpen) pooldb().start(true);
      }
      ~IOVImpl(){
	pooldb().commit();
      }
      cond::PoolTransaction& pooldb() { return connection.poolTransaction();}


      cond::Connection & connection;
      pool::Ref<cond::IOVSequence> iov;
    };

  }


  PoolTransaction *  IOVElementProxy::db() const {
    return connection() ? &connection()->poolTransaction() : (PoolTransaction *)(0);
  }


  IOVProxy::IterHelp::IterHelp(impl::IOVImpl & impl) :
    iov(*impl.iov), elem(&impl.connection){}
  

  void IOVElementProxy::set(IOVSequence const & v, int i) {
    if (i==v.iovs().size()) {
      set(0,0,"");
      return;
    }
    m_since =  v.iovs()[i].sinceTime();
    m_till  =  (i+1==v.iovs().size()) ? v.lastTill() : v.iovs()[i+1].sinceTime()-1;
    m_token = v.iovs()[i].wrapperToken();
  }



  IOVProxy::IOVProxy() : m_low(0), m_high(0){}
 
  IOVProxy::~IOVProxy() {}

  IOVProxy::IOVProxy(cond::Connection& conn,
		     const std::string & token, bool nolib, bool keepOpen) :
    m_iov(new impl::IOVImpl(conn,token,nolib,keepOpen)), m_low(0), m_high(size()){}


  void IOVProxy::setRange(cond::Time_t since, cond::Time_t  till) const {
    m_low=iov().find(since)-iov().iovs().begin();
    m_high=iov().find(till)-iov().iovs().begin();
    m_high=std::min(m_high+1,size());
  }

  void IOVProxy::head(int n) const {
    m_high = std::min(m_low+n,m_high);
  }

  void  IOVProxy::tail(int n) const {
    m_low = std::max(m_high-n,m_low);
  }


  IOVProxy::const_iterator IOVProxy::find(cond::Time_t time) const {
    int n = iov().find(time)-iov().iovs().begin();
    return (n<m_low || m_high<n ) ? 
      end() :  
      boost::make_transform_iterator(boost::counting_iterator<int>(n),
				     IterHelp(*m_iov));
  }


  int IOVProxy::size() const {
    return iov().iovs().size();
  }

  IOV const & IOVProxy::iov() const {
    return *(*m_iov).iov;
  }

  TimeType IOVProxy::timetype() const {
    return iov().timeType();     
  }

}

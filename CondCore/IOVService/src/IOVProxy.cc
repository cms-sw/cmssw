#include "CondCore/IOVService/interface/IOVProxy.h"
//#include "DataSvc/Ref.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"

#include "CondFormats/Common/interface/IOVSequence.h"

namespace cond {
  namespace impl {

    struct IOVImpl {
      IOVImpl(cond::DbSession& dbs,
	      const std::string & token,
	      bool /*nolib*/,
	      bool keepOpen) : poolDb(dbs), m_token(token), /*m_nolib(nolib),*/ m_keepOpen(keepOpen){
	refresh();
	if (m_keepOpen) {
	  if(poolDb.transaction().isActive() && !poolDb.transaction().isReadOnly())
	    poolDb.transaction().start(false);
	  else poolDb.transaction().start(true);
	}
      }
      
      void refresh() {
	cond::DbScopedTransaction transaction(poolDb);
	if(transaction.isActive() && !transaction.isReadOnly())
	  transaction.start(false);
	else transaction.start(true);
	iov = poolDb.getTypedObject<cond::IOVSequence>( m_token );
	transaction.commit();
	/*
	  if (!iov->iovs().empty() && !m_nolib) {
	  // load dict (change: use IOV metadata....)
	  std::string ptok = iov->iovs().front().wrapperToken();
	  cond::reflexTypeByToken(ptok);
          }
	*/
      }
      
      ~IOVImpl(){
	if (m_keepOpen) poolDb.transaction().commit();
      }
      
      cond::DbSession poolDb;
      boost::shared_ptr<cond::IOVSequence> iov;
      std::string m_token;
      //bool m_nolib;
      bool m_keepOpen;
    };
  }
  
  void IOVElementProxy::set(IOVSequence const & v, int ii) {
    size_t i =ii;
    if (i>=v.iovs().size()) {
      set(cond::invalidTime, cond::invalidTime,"");
      return;
    }
    m_since =  v.iovs()[i].sinceTime();
    m_till  =  (i+1==v.iovs().size()) ? v.lastTill() : v.iovs()[i+1].sinceTime()-1;
    m_token = v.iovs()[i].wrapperToken();
  }

  IOVProxy::IterHelp::IterHelp(impl::IOVImpl & impl) :
    iov(&(*impl.iov)), elem(impl.poolDb){}
  
  IOVProxy::IOVProxy() : m_low(0), m_high(0){}
 
  IOVProxy::~IOVProxy() {}

  IOVProxy::IOVProxy(cond::DbSession& dbSession,
                     const std::string & token,
                     bool nolib,
                     bool keepOpen)
    :m_iov(new impl::IOVImpl(dbSession,token,nolib,keepOpen)), m_low(0), m_high(size()){}


  bool IOVProxy::refresh() {
    int oldsize = size();
    m_iov->refresh();
   bool anew = oldsize<size();
    if (anew) m_high = size();  // FIXME
    return anew;
  }

  void IOVProxy::resetRange() const {
    m_low=0;
    m_high=size();
  }


  void IOVProxy::setRange(cond::Time_t since, cond::Time_t  till) const {
     m_low = (since<iov().iovs().front().sinceTime()) ? 0 :
      iov().find(since)-iov().iovs().begin();
    m_high=iov().find(till)-iov().iovs().begin();
    m_high=std::min(m_high+1,size());
  }


  //FIXME cannot be done twice....
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

  
  std::string 
  IOVProxy::payloadContainerName() const{
    // FIXME move to metadata
    std::string payloadtokstr=iov().iovs().front().wrapperToken();
    std::pair<std::string,int> oidData = parseToken( payloadtokstr );
    return oidData.first;
  }

  std::string 
  IOVProxy::comment() const{
    return iov().comment();
  }

  int 
  IOVProxy::revision() const{
    return iov().revision();
  }

  cond::Time_t IOVProxy::timestamp() const {
    return iov().timestamp();
  }

  cond::DbSession& IOVProxy::db() const {
    return m_iov->poolDb;
  }



}

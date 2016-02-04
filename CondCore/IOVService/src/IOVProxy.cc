#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"

#include "CondFormats/Common/interface/IOVSequence.h"

namespace cond {
  namespace impl {

    struct IOVImpl {
      explicit IOVImpl(cond::DbSession& dbs) : dbSession(dbs), m_token(""){
      }

      IOVImpl(cond::DbSession& dbs,
	      const std::string & token) : dbSession(dbs), m_token(token){
	refresh();
      }
      
      void refresh() {
	iov = dbSession.getTypedObject<cond::IOVSequence>( m_token );
        // loading the lazy-loading Queryable vector...
        iov->loadAll();
        //**** temporary for the schema transition
        if( dbSession.isOldSchema() ){
          PoolTokenParser parser(  dbSession.storage() ); 
          iov->swapTokens( parser );
        }
        //****
      }
      
      ~IOVImpl(){  
      }
      
      cond::DbSession dbSession;
      boost::shared_ptr<cond::IOVSequence> iov;
      std::string m_token;
    };
  }
  
  void IOVElementProxy::set(IOVSequence const & v, int ii) {
    size_t i =ii;
    if (i>=v.iovs().size()) {
      set(cond::invalidTime, cond::invalidTime,"");
      return;
    }
    m_token = v.iovs()[i].token();
    m_since =  v.iovs()[i].sinceTime();
    if(i+1==v.iovs().size()) {
      m_till = v.lastTill();
      return;
    }
    cond::UnpackedTime unpackedTime;
    cond::Time_t totalNanoseconds;
    cond::Time_t totalSecondsInNanoseconds;
    switch (v.timeType()) {
    case timestamp:
      //unpacking
      unpackedTime = cond::time::unpack(v.iovs()[i+1].sinceTime());
      //number of seconds in nanoseconds (avoid multiply and divide by 1e09)
      totalSecondsInNanoseconds = ((cond::Time_t)unpackedTime.first)*1000000000;
      //total number of nanoseconds
      totalNanoseconds = totalSecondsInNanoseconds + ((cond::Time_t)(unpackedTime.second));
      //now decrementing of 1 nanosecond
      totalNanoseconds--;
      //now repacking (just change the value of the previous pair)
      unpackedTime.first = (unsigned int) (totalNanoseconds/1000000000);
      unpackedTime.second = (unsigned int)(totalNanoseconds - (cond::Time_t)unpackedTime.first*1000000000);
      m_till = cond::time::pack(unpackedTime);
      break;
    default:
      m_till = v.iovs()[i+1].sinceTime()-1;
    }
  }

  IOVProxy::IterHelp::IterHelp(impl::IOVImpl & impl) :
    iov(&(*impl.iov)), elem(){}
  
  IOVProxy::IOVProxy() : m_low(0), m_high(0){}
 
  IOVProxy::~IOVProxy() {}

  IOVProxy::IOVProxy(cond::DbSession& dbSession)
    :m_iov(new impl::IOVImpl(dbSession)), m_low(0), m_high(0){}

  IOVProxy::IOVProxy(cond::DbSession& dbSession,
                     const std::string & token)
    :m_iov(new impl::IOVImpl(dbSession,token)), m_low(0), m_high(size()){}

  void IOVProxy::load( const std::string & token){
    m_iov->m_token = token;
    m_iov->refresh();
    m_high = size();
  }

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

  
  std::set<std::string> const& 
  IOVProxy::payloadClasses() const{
    return iov().payloadClasses();
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
    return m_iov->dbSession;
  }



}

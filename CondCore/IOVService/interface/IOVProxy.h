#ifndef CondCore_IOVService_IOVProxy_h
#define CondCore_IOVService_IOVProxy_h

#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/SequenceState.h"

#include <string>
#include "CondFormats/Common/interface/Time.h"
#include <boost/shared_ptr.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace cond {
  
  class IOVSequence;
  typedef  IOVSequence IOV;
  class PoolTransaction;
  class Connection;
  
  class IOVElementProxy {
  public:
    IOVElementProxy() : m_since(cond::invalidTime), m_till(cond::invalidTime), m_conn(0){}
    IOVElementProxy(Connection * iconn) : m_since(cond::invalidTime), m_till(cond::invalidTime), m_conn(iconn){}
    IOVElementProxy(cond::Time_t is,
	       cond::Time_t it,
	       std::string const& itoken,
	       Connection * iconn) :
      m_since(is), m_till(it), m_token(itoken), m_conn(iconn){}
    
    void set(cond::Time_t is,
	     cond::Time_t it,
	     std::string const& itoken ) {
      m_since=is; m_till=it; m_token=itoken;
    }
    
    void set(IOV const & v, int i);
    
    cond::Time_t since() const {return m_since;}
    cond::Time_t till() const {return m_till;}
    std::string const & token() const {return m_token;}
    std::string const & wrapperToken() const {return m_token;}
    Connection * connection() const { return m_conn;}
    PoolTransaction * db() const;

  private:
    cond::Time_t m_since;
    cond::Time_t m_till;
    std::string  m_token;
    Connection * m_conn;
  };
  
  
  namespace impl {
    struct IOVImpl;
  }
  
  /* IOV as the user wants it
   */
  class IOVProxy {
  public:
    
    IOVProxy();
    ~IOVProxy();
    
    IOVProxy(cond::Connection& conn,
	     const std::string & token, bool nolib, bool keepOpen);
    
    struct IterHelp {
      typedef IOVElementProxy result_type;
      IterHelp(impl::IOVImpl & in);
      
      result_type const & operator()(int i) const {
	elem.set(iov,i);
	return elem;
      } 
      
    private:
      IOV const & iov;
      mutable IOVElementProxy elem;
    };
    
    typedef boost::transform_iterator<IterHelp,boost::counting_iterator<int> > const_iterator;
    
    const_iterator begin() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(m_low),
					     IterHelp(*m_iov));
    }
    
    const_iterator end() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(m_high),
					     IterHelp(*m_iov));
    }
    
    // find in range...
    const_iterator find(cond::Time_t time) const;

    // limit range
    void setRange(cond::Time_t since, cond::Time_t  till) const;
    // limit to the first n 
    void head(int n) const;
    // limit to the last n
    void tail(int n) const;

    int size() const;
    IOV const & iov() const;
    TimeType timetype() const;
    std::string payloadContainerName() const;
    std::string comment() const;
    int revision() const;

    SequenceState state() const {
      return SequenceState(iov());
    }


 

    PoolTransaction & db() const;
 
    bool refresh();

  private:
    boost::shared_ptr<impl::IOVImpl> m_iov;
    mutable int m_low;
    mutable int m_high;
    
  };
}

#endif // CondCore_IOVService_IOVProxy_h

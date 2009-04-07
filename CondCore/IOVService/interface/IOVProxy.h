#ifndef CondCore_IOVService_IOVProxy_h
#define CondCore_IOVService_IOVProxy_h

#include "CondFormats/Common/interface/IOVSequence.h"
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
#include <boost/shared_ptr.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace cond {
  
  class IOVSequence;
  typedef  IOVSequence IOV;
  class PoolTransaction;
  
  class IOVElementProxy {
  public:
    IOVElementProxy() : m_since(0), m_till(0), m_db(0){}
    IOVElementProxy(PoolTransaction * idb) : m_since(0), m_till(0), m_db(idb){}
    IOVElementProxy(cond::Time_t is,
	       cond::Time_t it,
	       std::string const& itoken,
	       PoolTransaction * idb) :
      m_since(is), m_till(it), m_token(itoken), m_db(idb){}
    
    void set(cond::Time_t is,
	     cond::Time_t it,
	     std::string const& itoken ) {
      m_since=is; m_till=it; m_token=itoken;
    }
    
    void set(IOV const & v, int i);
    
    cond::Time_t since() const {return m_since;}
    cond::Time_t till() const {return m_till;}
    std::string const & wrapperToken() const {return m_token;}
    PoolTransaction * db() const { return m_db;}
  private:
    cond::Time_t m_since;
    cond::Time_t m_till;
    std::string  m_token;
    PoolTransaction * m_db;
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
    
    IOVProxy(cond::PoolTransaction& db,
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
    
    // limit range
    void setRange(cond::Time_t since, cond::Time_t  till) const;
    // limit to the first n 
    void head(int n) const;
    // limit to the last n
    void tail(int n) const;

    int size() const;
    IOV const & iov() const;
    TimeType timetype() const;
    
  private:
    boost::shared_ptr<impl::IOVImpl> m_iov;
    mutable int m_low;
    mutable int m_high;
    
  };
}

#endif // CondCore_IOVService_IOVProxy_h

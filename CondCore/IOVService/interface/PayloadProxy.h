#ifndef CondCore_IOVService_PayloadProxy_h
#define CondCore_IOVService_PayloadProxy_h

#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/PayloadRef.h"

namespace cond {

  /* get iov by name (key, tag, ...)

   */
  class CondGetter {
  public:
    virtual ~CondGetter(){}
    virtual IOVProxy get(std::string name) const=0;

  };

  /* implementation detail: 
    implements the not templated part...

   */
  class BasePayloadProxy {
  public:

    struct ObjId {
      cond::Time_t since;
      unsigned int oid1;
      unsigned int oid2;
    };
    typedef  std::vector<ObjId> ObjIds;
    struct Stats {
      int nProxy;
      int nRefresh;
      int nArefresh;
      int nMake;
      int nLoad;
      ObjIds ids;
    };

    // global stat
    static Stats gstats;
    // local stat
    Stats stats;

    // errorPolicy=true will throw on load, false will set interval and token to invalid
    BasePayloadProxy(cond::DbSession& session,
                     bool errorPolicy);

    // errorPolicy=true will throw on load, false will set interval and token to invalid
    BasePayloadProxy(cond::DbSession& session,
                     const std::string & token,
                     bool errorPolicy);

    void loadIov( const std::string iovToken );
    void loadTag( const std::string tag );
    
    virtual ~BasePayloadProxy();

    virtual void invalidateCache()=0;

    // current cached object token
    std::string const & token() const { return m_token;}
    
    // load Element valid at time
    cond::ValidityInterval loadFor(cond::Time_t time);
    
    // load nth Element (to be used in simple iterators...)
    cond::ValidityInterval loadFor(size_t n);

    // find ad return interval (does not load)
    cond::ValidityInterval setIntervalFor(cond::Time_t time);
    
    // load element if interval is valid
    void make();

    bool isValid() const;

    TimeType timetype() const { return m_iov.timetype();}

    IOVProxy const & iov() const { return m_iov;}

    virtual void loadMore(CondGetter const &){}

    // reload the iov return true if size has changed
    bool refresh();
    bool refresh( cond::DbSession& newSession );


  private:
    virtual bool load(cond::DbSession& session, std::string const & token) =0;   


  protected:
    bool m_doThrow;
    IOVProxy m_iov;
    IOVElementProxy m_element;
    mutable DbSession m_session;

  protected:
    // current loaded payload
    std::string  m_token;

  };


  /* proxy to the payload valid at a given time...

   */
  template<typename DataT>
  class PayloadProxy : public BasePayloadProxy {
  public:
 
    PayloadProxy(cond::DbSession& session,
                 bool errorPolicy,
                 const char * source=0) :
      BasePayloadProxy(session, errorPolicy) {}

    PayloadProxy(cond::DbSession& session,
                 const std::string & token,
                 bool errorPolicy,
                 const char * source=0) :
      BasePayloadProxy(session, token, errorPolicy) {}
    
    virtual ~PayloadProxy(){}

    // dereference (does not load)
    const DataT & operator()() const {
      return (*m_data); 
    }
        
    virtual void invalidateCache() {
      m_data.clear();
      m_token.clear(); // in base....
    }


  protected:
    virtual bool load(cond::DbSession& session, std::string const & itoken) {
      return m_data.load(session,itoken);
    }

  private:
     cond::PayloadRef<DataT> m_data;
  };

}
#endif // CondCore_IOVService_PayloadProxy_h

#ifndef CondCore_IOVService_PayloadProxy_h
#define CondCore_IOVService_PayloadProxy_h

#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/PayloadRef.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"

namespace pool{
  class IDataSvc;
}


namespace cond {

  /* get iov by name (key, tag...)

   */
  class CondGetter {
  public:
    virtual ~CondGetter();
    IOVProxy get(std::string name) const=0;

  };

  /* implementation detail: 
    implements the not templated part...

   */
  class BasePayloadProxy {
  public:
    // errorPolicy=true will throw on load, false will set interval and token to invalid
    BasePayloadProxy(cond::Connection& conn,
		     const std::string & token, bool errorPolicy);
    
    virtual~BasePayloadProxy();

    virtual void invalidateCache()=0;

    // load Element valid at time
    void loadFor(cond::Time_t time);

    // find ad return interval (does not load)
    cond::ValidityInterval setIntervalFor(cond::Time_t time);
    
    // load element if interval is valid
    void make();

    bool isValid() const;

    TimeType timetype() const { return m_iov.timetype();}

    IOVProxy const & iov() const { return m_iov;}

  private:
    virtual bool load(pool::IDataSvc * svc, std::string const & token) =0;   


  protected:
    bool m_doThrow;
    IOVProxy m_iov;
    IOVElementProxy m_element;
  };


  /* proxy to the payload valid at a given time...

   */
  template<typename DataT>
  class PayloadProxy : public BasePayloadProxy {
  public:
 
    PayloadProxy(cond::Connection& conn,
		 const std::string & token, bool errorPolicy) :
      BasePayloadProxy(conn, token, errorPolicy) {}
    
    virtual ~PayloadProxy(){}

    // dereference (does not load)
    const DataT & operator()() const {
      (*m_data); 
    }
        
    virtual void invalidateCache() {
      m_data.clear();
    }

  protected:
    virtual bool load(pool::IDataSvc * svc, std::string const & token) {
      return m_data.load(svc,token);

  private:
     cond::PayloadRef<DataT> m_data;
  };

}
#endif // CondCore_IOVService_PayloadProxy_h

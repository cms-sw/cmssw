#ifndef CondCore_IOVService_PayloadProxy_h
#define CondCore_IOVService_PayloadProxy_h

#include "CondCore/IOVService/interface/IOVProxy.h"
#include "DataSvc/Ref.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"

namespace pool{
  class IDataSvc;
}


namespace cond {

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
    typedef cond::DataWrapper<DataT> DataWrapper;

    PayloadProxy(cond::Connection& conn,
		 const std::string & token, bool errorPolicy) :
      BasePayloadProxy(conn, token, errorPolicy), old(false){}
    
    virtual ~PayloadProxy(){}

    // dereference (does not load)
    const DataT & operator()() const {
      return old ? *m_OldData : m_data->data(); 
    }
    
    
    virtual void invalidateCache() {
      m_data.clear();
      m_OldData.clear();
    }

  private:
    virtual bool load(pool::IDataSvc * svc, std::string const & token) {
      old = false;
      invalidateCache();
      bool ok = false;
      // try wrapper, if not try plain
      pool::Ref<DataWrapper> ref(svc,token);
      if (ref) {
	m_data.copyShallow(ref);
	m_data->data();
	ok= true;
      } else {
	pool::Ref<DataT> refo(svc,token);
	if (refo) {
	  old = true;
	  m_OldData.copyShallow(refo);
	  ok =  true;
	}
      }
      return ok;
    }
    
    
  private:
    bool old;
    pool::Ref<DataWrapper> m_data;
    // Backward compatibility
    pool::Ref<DataT> m_OldData;
  };

}
#endif // CondCore_IOVService_PayloadProxy_h

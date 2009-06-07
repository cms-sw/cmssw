#ifndef CondCore_IOVService_KeyListProxy_h
#define CondCore_IOVService_KeyListProxy_h

#include "CondCore/IOVService/interface/PayloadProxy.h"
#include "CondCore/IOVService/interface/KeyList.h"


namespace cond {

  template<> class PayloadProxy<cond::KeyList> : public PayloadProxy<vector<cond::Time_t> > {
  public:
    typedef vector<cond::Time_t> DataT;
    typedef Payload<DataT> super;

    PayloadProxy(cond::Connection& conn,
		 const std::string & token, bool errorPolicy) :
      super(conn, token, errorPolicy) {}
    
    virtual ~PayloadProxy(){}

    // dereference (does not load)
    const cond::KeyList & operator()() const {
      return me; 
    }
        
    virtual void invalidateCache() {
      super::invalidateCache();
    }

    virtual void loadMore(CondGetter const & getter){
      me.init(getter(name));
    }


  protected:
    virtual bool load(pool::IDataSvc * svc, std::string const & token) {
     bool ok = super::load();
      me.load(super::operator()());
      return ok;
    }

  private:

    std::string m_name

    KeyList me;

  };
}
#endif


}

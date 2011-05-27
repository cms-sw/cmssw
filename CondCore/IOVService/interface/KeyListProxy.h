#ifndef CondCore_IOVService_KeyListProxy_h
#define CondCore_IOVService_KeyListProxy_h

#include "CondCore/IOVService/interface/PayloadProxy.h"
#include "CondCore/IOVService/interface/KeyList.h"
#include <vector>
#include <string>

namespace cond {

  template<> class PayloadProxy<cond::KeyList> : public PayloadProxy<std::vector<cond::Time_t> > {
  public:
    typedef std::vector<cond::Time_t> DataT;
    typedef PayloadProxy<DataT> super;

    PayloadProxy(cond::DbSession& session,
		 bool errorPolicy, const char * source=0) :
      super(session, errorPolicy) {
      m_name = source;
    }

    PayloadProxy(cond::DbSession& session,
		 const std::string & token, bool errorPolicy, const char * source=0) :
      super(session, token, errorPolicy) {
      m_name = source;
    }
    
    virtual ~PayloadProxy(){}

    // dereference (does not load)
    const cond::KeyList & operator()() const {
      return me; 
    }
        
    virtual void invalidateCache() {
      super::invalidateCache();
    }

    virtual void loadMore(CondGetter const & getter){
      me.init(getter.get(m_name));
    }


  protected:
    virtual bool load( DbSession& sess, std::string const & token) {
      bool ok = super::load(sess, token);
      me.load(super::operator()());
      return ok;
    }

  private:

    std::string m_name;

    KeyList me;

  };
}
#endif

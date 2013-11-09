#ifndef CondCore_CondDB_KeyListProxy_h
#define CondCore_CondDB_KeyListProxy_h

#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include <vector>
#include <string>

namespace cond {

  using KeyList = cond::ora_wrapper::KeyList;

  namespace db {
    template<> class PayloadProxy<cond::KeyList> : public PayloadProxy<std::vector<cond::Time_t> > {
    public:
      typedef std::vector<cond::Time_t> DataT;
      typedef PayloadProxy<DataT> super;

    
      PayloadProxy( Session& session, const char * source=0 ) :
	super( session ),
	m_keyList( session ) {
	if( source ) m_name = source;
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
      	me.init(getter.getTag(m_name));
      }


    protected:
      virtual bool loadPayload() {
	bool ok = super::loadPayload();
	me.load(super::operator()());
	return ok;
      }

  private:

    std::string m_name;
    KeyList m_keyList;

  };
}
#endif

#ifndef CondCore_CondDB_KeyListProxy_h
#define CondCore_CondDB_KeyListProxy_h

#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include <vector>
#include <string>

namespace cond {

  namespace persistency {
    template<> class PayloadProxy<cond::persistency::KeyList> : public PayloadProxy<std::vector<cond::Time_t> > {
    public:
      typedef std::vector<cond::Time_t> DataT;
      typedef PayloadProxy<DataT> super;

    
      explicit PayloadProxy( const char * source=0 ) :
	super( source ),
	m_keyList() {
	if( source ) m_name = source;
      }

      virtual ~PayloadProxy(){}

      // dereference (does not load)
      const KeyList & operator()() const {
	return m_keyList; 
      }
        
      virtual void invalidateCache() {
	super::invalidateCache();
      }

      virtual void loadMore(CondGetter const & getter){
      	m_keyList.init(getter.get(m_name));
      }


    protected:
      virtual void loadPayload() {
	super::loadPayload();
	m_keyList.load(super::operator()());
      }

    private:
      
      std::string m_name;
      KeyList m_keyList;

    };
  }

}
#endif

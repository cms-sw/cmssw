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

    
      explicit PayloadProxy( const char * source=nullptr ) :
	super( source ),
	m_keyList() {
	if( source ) m_name = source;
      }

      ~PayloadProxy() override{}

      // dereference (does not load)
      const KeyList & operator()() const {
	return m_keyList; 
      }
        
      void invalidateCache() override {
	super::invalidateCache();
      }

      void loadMore(CondGetter const & getter) override{
      	m_keyList.init(getter.get(m_name));
      }


    protected:
      void loadPayload() override {
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

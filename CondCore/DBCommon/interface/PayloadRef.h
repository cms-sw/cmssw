#ifndef CondCore_DBCommon_PayloadRef_h
#define CondCore_DBCommon_PayloadRef_h

#include "DataSvc/Ref.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"

namespace pool{
  class IDataSvc;
}


namespace cond {

  /* manages various types of wrappers...
   */
  template<typename DataT>
  class PayloadRef {
  public:
    PayloadRef() : old(false){}
    ~PayloadRef(){}

   // dereference (does not re-load)
    const DataT & operator*() const {
      return old ? *m_OldData : m_data->data(); 
    }
    
    
   void clear() {
      m_data.clear();
      m_OldData.clear();
    }

    
    bool load(pool::IDataSvc * svc, std::string const & token) {
      old = false;
      clear();
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

#ifndef CondCore_DBCommon_PayloadRef_h
#define CondCore_DBCommon_PayloadRef_h

#include "DataSvc/Ref.h"

namespace pool{
  class IDataSvc;
}

namespace cond {

  /* manages various types of wrappers...
   */
  template<typename DataT>
  class PayloadRef {
  public:
 
    PayloadRef(){}
    ~PayloadRef(){}
    
    // dereference (does not re-load)
    const DataT & operator*() const {
      *m_Data; 
    }
    
    void clear() {
      m_Data.clear();
    }
    
    
    bool load(pool::IDataSvc * svc, std::string const & itoken) {
      clear();
      bool ok = false;
     
      pool::Ref<DataT> refo(svc,itoken);
      if (refo) {
	m_Data.copyShallow(refo);
	ok =  true;
      }
      return ok;
    }
    
    
  private:
    pool::Ref<DataT> m_Data;
  };
  
}
#endif // CondCore_PayloadProxy_h

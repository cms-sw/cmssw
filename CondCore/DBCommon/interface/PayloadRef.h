#ifndef CondCore_DBCommon_PayloadRef_h
#define CondCore_DBCommon_PayloadRef_h
#include "CondCore/DBCommon/interface/DbTransaction.h"

namespace cond {

  /* manages various types of wrappers...
   */
  template<typename DataT>
  class PayloadRef {
  public:

    PayloadRef() {}
    ~PayloadRef(){}
    
    // dereference (does not re-load)
    const DataT & operator*() const {
      return *m_Data; 
    }
    
    void clear() {
      m_Data.reset();
    }
    
    
    bool load( DbSession& dbSess, std::string const & itoken) {
      clear();
      bool ok = false;
      // is it ok to open a transaction here? or could be at higher level?
      boost::shared_ptr<DataT> tmp = dbSess.getTypedObject<DataT>( itoken );
      if (tmp.get()) {
	m_Data = tmp;
	ok =  true;
      }
      return ok;
    }
    
    
  private:
    boost::shared_ptr<DataT> m_Data;
  };
  
}
#endif // CondCore_PayloadProxy_h

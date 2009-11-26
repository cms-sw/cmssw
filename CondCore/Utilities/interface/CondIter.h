#ifndef CondIter_CondIter_h
#define CondIter_CondIter_h


#include "CondCore/Utilities/interface/CondBasicIter.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"




template <class T>
class CondIter : public  CondBasicIter{
  
  protected:
  virtual bool load(pool::IDataSvc * svc, std::string const & itoken) {
    return m_data.load(svc,itoken);
  }

private:
  cond::PayloadRef<DataT> m_data;
 
public:
  
  
  CondIter(){}
  ~CondIter(){}
  
  
  
  
 
  /**
     Obtain the pointer to an object T. If it is the last T the method returns a null pointer.
  */ 
  
  
  T const * next() {
    bool ok=false;;
    if (!initialized) ok =init();
    else ok = forward();
    if (!ok) return 0;
    ok = make();
    if (!ok) return 0;
    return &(*m_data()); 
};




#endif


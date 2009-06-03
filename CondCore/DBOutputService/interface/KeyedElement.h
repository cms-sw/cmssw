#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/GenericSummary.h"
#include "CondFormats/Common/interface/hash64.h"

namespace cond{

  /* encapsulate data to write a keyed element in an IOVSequence
   */
  class KeyedElement {

  public:
   KeyedElement(BaseKeyed * obj, std::string key) : 
      m_obj(obj), 
      m_sum(new cond::GenericSummary(key)), 
      m_key(cond::hash64(&key[0],key.size(),0) {}

    BaseKeyed * m_obj;
    cond::Summary * m_sum;
    cond::Time_t m_key;
  };
  
  
}

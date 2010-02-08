#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/GenericSummary.h"
#include "CondFormats/Common/interface/hash64.h"
#inlcude <sstream>

namespace cond{

  /* encapsulate data to write a keyed element in an IOVSequence
   */
  class KeyedElement {

  public:
    // constructor from int key
   KeyedElement(BaseKeyed * obj, cond::Time_t key) : 
     m_obj(obj), 
     m_sum(0), 
     m_key(key) {
     std::ostringstream ss; ss << key;
     m_sum (new cond::GenericSummary(ss.str())); 
   }

    // constructor from ascii key
   KeyedElement(BaseKeyed * obj, std::string key) : 
      m_obj(obj), 
      m_sum(new cond::GenericSummary(key)), 
      m_key(convert(key)) {}
    
    static cond::Time_t convert(std::string key) {
      return cond::hash64( (unsigned char*)(&key[0]),key.size(),0);
    }

    BaseKeyed * m_obj;
    cond::Summary * m_sum;
    cond::Time_t m_key;
  };
  
  
}

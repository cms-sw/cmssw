#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/hash64.h"
#include <sstream>

namespace cond{

  /* encapsulate data to write a keyed element in an IOVSequence
   */
  class KeyedElement {

  public:
    // constructor from int key
   KeyedElement(BaseKeyed * obj, cond::Time_t key) : 
     m_obj(obj), 
     m_skey(""),
     m_key(key) {
     std::ostringstream ss; ss << key;
     m_skey  = ss.str();
     (*obj).setKey(m_skey); 
   }

    // constructor from ascii key
   KeyedElement(BaseKeyed * obj, std::string key) : 
      m_obj(obj), 
      m_skey(key),
      m_key(convert(key)) {
     (*obj).setKey(m_skey);
   }
    
    static cond::Time_t convert(std::string key) {
      return cond::hash64( (unsigned char*)(&key[0]),key.size(),0);
    }

    BaseKeyed * m_obj;
    std::string  m_skey;
    cond::Time_t m_key;
  };
  
  
}

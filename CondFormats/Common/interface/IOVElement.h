#ifndef Cond_IOVElement_h
#define Cond_IOVElement_h
#include "CondFormats/Common/interface/Time.h"
#include <string>

namespace cond {

  /** Element of an IOV Sequence
      include the since time, token to the wrapper
   */
  class IOVElement {
  public:
    IOVElement(){}
    // for comparisons
    IOVElement(cond::Time_t it) : m_sinceTime(it){}

    IOVElement(cond::Time_t it,
	       std::string const& iwrapper):
      m_sinceTime(it),m_wrapper(iwrapper){}

    cond::Time_t sinceTime() const {return m_sinceTime;}
    std::string const & token()  const {return m_wrapper;}
    std::string const & wrapperToken()  const {return m_wrapper;}

    bool operator==(IOVElement const & rh) const {
      return sinceTime()==rh.sinceTime()
	&&  wrapperToken()==rh.wrapperToken();
    }

    
  private:
    cond::Time_t m_sinceTime;
    std::string m_wrapper;
    
  };
  
  
} // ns cond


#endif

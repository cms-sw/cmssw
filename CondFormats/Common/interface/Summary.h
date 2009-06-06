#ifndef Cond_Summary_h
#define Cond_Summary_h

#include <string>
#include <iosfwd>

namespace cond {
  
  /** Base class for summary of condition payoad
  */
  class Summary {
  public:
    
    Summary();
    virtual ~Summary();
    
     // short message (just content to be used in a table)
    virtual void shortMessage(std::ostream & os) const=0;
    
    // long message (ot be used in pop-up, single views)
    virtual void longMessage(std::ostream & os) const=0;
    
    
  };
  
}

inline std::ostream & operator<<(std::ostream & os, cond::Summary const & s) {
  s.shortMessage(os); return os;
}

#endif

#ifndef Cond_GenericSummary_h
#define Cond_GenericSummary_h

#include "CondFormats/Common/interface/Summary.h"

namespace cond {

  /** Short summary of condition payoad
  */
  class GenericSummary : public Summary {
  public:
    
    GenericSummary();
    virtual ~GenericSummary();
    
    //
    explicit GenericSummary(std::string const & s);
    
    // short message (just content to be used in a table)
    virtual void shortMessage(std::ostream & os) const;
    
    // long message (to be used in pop-up, single views)
    virtual void longMessage(std::ostream & os) const;
    
    
  private:
    
    std::string m_me;
    
  };
  

}


#endif

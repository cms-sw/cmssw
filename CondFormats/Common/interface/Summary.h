#ifndef Cond_Summary_h
#define Cond_Summary_h

#include <string>
#include <iosfwd>

namespace cond {

  /** Short summary of condition payoad
      FIXME in future polymorfic
  */
  class Summary {
  public:
    
    Summary();
    virtual ~Summary();
    
    //
    explicit Summary(std::string const & s);
    
    // short message (just content to be used in a table)
    virtual void shortMessage(std::ostream & os) const;
    
    // long message (ot be used in pop-up, single views)
    virtual void longMessage(std::ostream & os) const;
    
    
private:
    
    std::string m_me;
    
  };
  
}

inline std::ostream & operator<<(std::ostream & os, cond::Summary const & s) {
  return s.shortMessage(os);
}

#endif

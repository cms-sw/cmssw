#include "CondFormats/Common/interface/Summary.h"


namespace cond {
  Summary(){}

  ~Summary(){}
    
  Summary(std::string const & s) : m_me(s){}
    
  void shortMessage(ostream & os) const {
    os << m_me;
  }
    
  void longMessage(ostream & os) const {
    os << m_me;
  }

}
   

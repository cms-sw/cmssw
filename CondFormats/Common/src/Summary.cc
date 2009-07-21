#include "CondFormats/Common/interface/Summary.h"
#include<ostream>

namespace cond {
  Summary::Summary(){}

  Summary::~Summary(){}
    
  Summary::Summary(std::string const & s) : m_me(s){}
    
  void Summary::shortMessage(std::ostream & os) const {
    os << m_me;
  }
    
  void Summary::longMessage(std::ostream & os) const {
    os << m_me;
  }

}
   

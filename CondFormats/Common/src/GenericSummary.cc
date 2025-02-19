#include "CondFormats/Common/interface/GenericSummary.h"
#include<ostream>

namespace cond {
  GenericSummary::GenericSummary(){}

  GenericSummary::~GenericSummary(){}
    
  GenericSummary::GenericSummary(std::string const & s) : m_me(s){}
    
  void GenericSummary::shortMessage(std::ostream & os) const {
    os << m_me;
  }
    
  void GenericSummary::longMessage(std::ostream & os) const {
    os << m_me;
  }

}
   

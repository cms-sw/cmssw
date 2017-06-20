#ifndef CommonTools_UtilAlgos_TwoObjectCalculator_H
#define CommonTools_UtilAlgos_TwoObjectCalculator_H

#include <string>
#include <cmath>

struct CosDphiCalculator {
  template <typename LHS, typename RHS > double operator()( const LHS & lhs, const RHS & rhs){
    double cdphi = cos(lhs.phi()-rhs.phi());
    return cdphi;    
  }  
  static std::string calculationType(){ return "CosDphiCalculator";}
  static std::string description() { return " calculate cos(Delta Phi(Obj1, Obj2))";}
};

#endif

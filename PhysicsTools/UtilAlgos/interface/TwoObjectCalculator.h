#ifndef PhysicsTools_UtilAlgos_TwoObjectCalculator_H
#define PhysicsTools_UtilAlgos_TwoObjectCalculator_H

struct CosDphiCalculator {
  template <typename LHS, typename RHS > double operator()( const LHS & lhs, const RHS & rhs){
    double cdphi = cos(lhs.phi()-rhs.phi());
    return cdphi;    
  }  
  static std::string calculationType(){ return "CosDphiCalculator";}
};

#endif

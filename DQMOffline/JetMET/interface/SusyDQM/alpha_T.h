#ifndef ALPHA_T_H
#define ALPHA_T_H

#include <cmath>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>

struct alpha_T {
  
  template<class LVs>
  double operator()(const LVs& p4s) const {  typedef typename LVs::value_type LorentzV;
    if( p4s.size() < 2 ) return 0;
    
    std::vector<double> pTs;  
    transform( p4s.begin(), p4s.end(), back_inserter(pTs), std::mem_fun_ref(&LorentzV::Pt));
    
    const double DsumPT = minimum_deltaSumPT( pTs );
    const double sumPT = accumulate( pTs.begin(), pTs.end(), double(0) );
    const LorentzV sumP4 = accumulate( p4s.begin(), p4s.end(), LorentzV() );
    
    return 0.5 * ( sumPT - DsumPT ) / sqrt( sumPT*sumPT - sumP4.Perp2() );
  }
  
  static double minimum_deltaSumPT(const std::vector<double>& pTs) {
    std::vector<double> diff( 1<<(pTs.size()-1) , 0. );
    for(unsigned i=0; i < diff.size(); i++)
      for(unsigned j=0; j < pTs.size(); j++)
	diff[i] += pTs[j] * ( 1 - 2 * (int(i>>j)&1) ) ;
    
    return fabs( *min_element( diff.begin(), diff.end(),
                [](auto x, auto y){return fabs(x) < fabs(y);} ) );
  }
  
};

#endif

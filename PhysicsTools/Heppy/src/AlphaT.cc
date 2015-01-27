#include "PhysicsTools/Heppy/interface/AlphaT.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace heppy {

double AlphaT::getAlphaT( const std::vector<double>& et,
			  const std::vector<double>& px,
			  const std::vector<double>& py ){
    
    // Momentum sums in transverse plane
    const double sum_et = accumulate( et.begin(), et.end(), 0. );
    const double sum_px = accumulate( px.begin(), px.end(), 0. );
    const double sum_py = accumulate( py.begin(), py.end(), 0. );
    
    // Minimum Delta Et for two pseudo-jets
    double min_delta_sum_et = -1.;
    for ( unsigned i=0; i < unsigned(1<<(et.size()-1)); i++ ) { //@@ iterate through different combinations
      double delta_sum_et = 0.;
      std::vector<bool> jet;
      for ( unsigned j=0; j < et.size(); j++ ) { //@@ iterate through jets
	delta_sum_et += et[j] * ( 1 - 2 * (int(i>>j)&1) ); 
      }
      if ( ( fabs(delta_sum_et) < min_delta_sum_et || min_delta_sum_et < 0. ) ) {
	min_delta_sum_et = fabs(delta_sum_et);
      }
    }
    if ( min_delta_sum_et < 0. ) { return 0.; }
    
    // Alpha_T
    return ( 0.5 * ( sum_et - min_delta_sum_et ) / sqrt( sum_et*sum_et - (sum_px*sum_px+sum_py*sum_py) ) );

  }
}

#include "PhysicsTools/Heppy/interface/AlphaT.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace heppy {

double AlphaT::getAlphaT( const std::vector<double>& et,
			  const std::vector<double>& px,
			  const std::vector<double>& py,
			  std::vector<int> * jet_pseudoFlag,
			  double& minDeltaHT){
   
    // Clear pesudo-jet container
    if (jet_pseudoFlag){
      jet_pseudoFlag->clear();
      jet_pseudoFlag->resize(et.size());
    }

    // Initialization of DeltaHT to overwrite the previous values stored
    minDeltaHT = 0.;

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
	if (jet_pseudoFlag){
	// if (!jet_pseudoFlag.empty()){
          for (unsigned int j = 0; j < et.size(); ++j){
            //bool testBool = ((i & (1U << j)) == 0);
	    if (((i & (1U << j)) == 0)) (*jet_pseudoFlag)[j] = 1;
	    else (*jet_pseudoFlag)[j] = 0;
	  }
        }
      }
    }
    
    if ( min_delta_sum_et < 0. ) { return 0.; }
   
    minDeltaHT = min_delta_sum_et;

    // Alpha_T
    return ( 0.5 * ( sum_et - min_delta_sum_et ) / sqrt( sum_et*sum_et - (sum_px*sum_px+sum_py*sum_py) ) );

  }
}

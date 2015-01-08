#include "HLTrigger/JetMET/interface/AlphaT.h"

double AlphaT::value_(std::vector<bool> * jet_sign) const {

  // Clear pseudo-jet container
  if (jet_sign) {
    jet_sign->clear();
    jet_sign->resize(et_.size());
  }

  // check the size of the input collection
  if (et_.size() == 0)
    // empty jet collection, return AlphaT = 0
    return 0.;

  if (et_.size() > (unsigned int) std::numeric_limits<unsigned int>::digits)
    // too many jets, return AlphaT = a very large number
    return std::numeric_limits<double>::max(); 

  // Momentum sums in transverse plane
  const double sum_et = std::accumulate( et_.begin(), et_.end(), 0. );
  const double sum_px = std::accumulate( px_.begin(), px_.end(), 0. );
  const double sum_py = std::accumulate( py_.begin(), py_.end(), 0. );

  // Minimum Delta Et for two pseudo-jets
  double min_delta_sum_et = sum_et;

  if(setDHtZero_){
    min_delta_sum_et = 0.;
  }else{
    for (unsigned int i = 0; i < (1U << (et_.size() - 1)); i++) { //@@ iterate through different combinations
      double delta_sum_et = 0.;
      for (unsigned int j = 0; j < et_.size(); ++j) { //@@ iterate through jets
        if (i & (1U << j))
          delta_sum_et -= et_[j];
        else
          delta_sum_et += et_[j];
      }
      delta_sum_et = std::abs(delta_sum_et);
      if (delta_sum_et < min_delta_sum_et) {
        min_delta_sum_et = delta_sum_et;
        if (jet_sign) {
          for (unsigned int j = 0; j < et_.size(); ++j)
            (*jet_sign)[j] = ((i & (1U << j)) == 0);
        }
      }
    }
  }
  // Alpha_T
  return (0.5 * (sum_et - min_delta_sum_et) / sqrt( sum_et*sum_et - (sum_px*sum_px+sum_py*sum_py) ));  
}

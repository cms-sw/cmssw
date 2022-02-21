#include "TTreeValidation.h"

namespace mkfit {

  Validation* Validation::make_validation(const std::string& fileName, const TrackerInfo* trk_info) {
#ifndef NO_ROOT
    if (Config::sim_val_for_cmssw || Config::sim_val || Config::fit_val || Config::cmssw_val) {
      return new TTreeValidation(fileName, trk_info);
    }
#endif
    return new Validation();
  }

  Validation::Validation() {}

}  // end namespace mkfit

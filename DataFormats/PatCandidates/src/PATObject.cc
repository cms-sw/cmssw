#include "DataFormats/PatCandidates/interface/PATObject.h"

namespace pat {
  const reco::CandidatePtrVector& get_empty_cpv() {
    static const reco::CandidatePtrVector EMPTY_CPV;
    return EMPTY_CPV;
  }

  const std::string& get_empty_str() {
    static const std::string EMPTY_STR;
    return EMPTY_STR;
  }
}  // namespace pat

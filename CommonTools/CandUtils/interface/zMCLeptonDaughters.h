#ifndef CommonTools_CandUtils_zMCLeptonDaughters_h
#define CommonTools_CandUtils_zMCLeptonDaughters_h
#include <utility>

#include "DataFormats/Candidate/interface/CandidateOnlyFwd.h"

namespace reco {
  std::pair<const Candidate *, const Candidate *> zMCLeptonDaughters(const Candidate &z, int leptonPdgId);
}  // namespace reco

#endif

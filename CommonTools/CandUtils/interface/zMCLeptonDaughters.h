#ifndef CommonTools_CandUtils_zMCLeptonDaughters_h
#define CommonTools_CandUtils_zMCLeptonDaughters_h
#include <utility>

namespace reco {
  class Candidate;
  std::pair<const Candidate*, const Candidate *>
    zMCLeptonDaughters(const Candidate & z, int leptonPdgId );
}

#endif

#ifndef CandMCTag_H
#define CandMCTag_h

#include "DataFormats/Candidate/interface/Candidate.h"

namespace CandMCTagUtils {

  std::vector<const reco::Candidate *> getAncestors(const reco::Candidate &c);
  bool hasBottom(const reco::Candidate &c);
  bool hasCharm(const reco::Candidate &c);
  bool isParton(const reco::Candidate &c);
  bool isLightParton(const reco::Candidate &c);

}
#endif

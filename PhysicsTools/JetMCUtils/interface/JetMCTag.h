#ifndef JetMCTag_H
#define JetMCTag_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"

namespace JetMCTagUtils {

  double EnergyRatioFromBHadrons(const reco::Candidate &c);
  double EnergyRatioFromCHadrons(const reco::Candidate &c);
  bool   decayFromBHadron(const reco::Candidate &c);
  bool   decayFromCHadron(const reco::Candidate &c);
  std::string genTauDecayMode(const reco::CompositePtrCandidate &c);

}
#endif

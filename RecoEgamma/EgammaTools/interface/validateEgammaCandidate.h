#ifndef __RecoEgamma_EgammaTools_EgammaCandidateValidation_H__
#define __RecoEgamma_EgammaTools_EgammaCandidateValidation_H__

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

namespace egammaTools {

  void validateGsfElectron(reco::GsfElectron const& electron);

  template <class Candidate>
  void validateEgammaCandidate(Candidate const& candidate) {
    if constexpr (std::is_same<Candidate, reco::GsfElectron>()) {
      validateGsfElectron(candidate);
    }
  }

}  // namespace egammaTools

#endif

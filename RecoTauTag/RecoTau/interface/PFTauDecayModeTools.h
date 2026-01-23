#ifndef RecoTauTag_RecoTau_PFTauDecayModeTools_h
#define RecoTauTag_RecoTau_PFTauDecayModeTools_h

/*
 * Tools to help deal with the PFTau::hadronicDecayMode
 * defintion.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/GenJetFwd.h"

namespace reco {
  namespace tau {

    /// Reverse mapping of decay modes into multiplicities
    unsigned int chargedHadronsInDecayMode(PFTau::hadronicDecayMode mode);

    unsigned int piZerosInDecayMode(PFTau::hadronicDecayMode mode);

    PFTau::hadronicDecayMode translateDecayMode(unsigned int nCharged, unsigned int nPiZero);

    /// Convert a genTau decay mode string ('oneProng0Pi0') to the RECO enum
    PFTau::hadronicDecayMode translateGenDecayModeToReco(const std::string& genName);

    /// Convert a RECO enum decay mode to a string ('oneProng0Pi0')
    std::string translateRecoDecayModeToGen(PFTau::hadronicDecayMode decayMode);

    PFTau::hadronicDecayMode getDecayMode(const reco::GenJet* genJet);

  }  // namespace tau
}  // namespace reco

#endif

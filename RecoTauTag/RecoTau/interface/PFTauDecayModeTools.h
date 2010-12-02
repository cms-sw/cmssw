#ifndef RecoTauTag_RecoTau_RecoTauCommonUtilities_h
#define RecoTauTag_RecoTau_RecoTauCommonUtilities_h

/*
 * Tools to help deal with the PFTau::hadronicDecayMode
 * defintion.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include "DataFormats/TauReco/interface/PFTau.h"

namespace reco { namespace tau {

/// Reverse mapping of decay modes into multiplicities
unsigned int chargedHadronsInDecayMode(
    PFTau::hadronicDecayMode mode);

unsigned int piZerosInDecayMode(
    PFTau::hadronicDecayMode mode);

PFTau::hadronicDecayMode translateDecayMode(
    unsigned int nCharged, unsigned int nPiZero);

}}

#endif

#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"

namespace reco { namespace tau {

unsigned int chargedHadronsInDecayMode(PFTau::hadronicDecayMode mode) {
   int modeAsInt = static_cast<int>(mode);
   return (modeAsInt / PFTau::kOneProngNPiZero) + 1;
}

unsigned int piZerosInDecayMode(PFTau::hadronicDecayMode mode) {
   int modeAsInt = static_cast<int>(mode);
   return (modeAsInt % PFTau::kOneProngNPiZero);
}

PFTau::hadronicDecayMode translateDecayMode(
    unsigned int nCharged, unsigned int nPiZeros) {
   // If no tracks exist, this is definitely not a tau!
   if(!nCharged) return PFTau::kNull;
   // Find the maximum number of PiZeros our parameterization can hold
   const unsigned int maxPiZeros = PFTau::kOneProngNPiZero;
   // Determine our track index
   unsigned int trackIndex = (nCharged-1)*(maxPiZeros+1);
   // Check if we handle the given number of tracks
   if(trackIndex >= PFTau::kRareDecayMode) return PFTau::kRareDecayMode;

   nPiZeros = (nPiZeros <= maxPiZeros) ? nPiZeros : maxPiZeros;
   return static_cast<PFTau::hadronicDecayMode>(trackIndex + nPiZeros);
}

}} // end namespace reco::tau

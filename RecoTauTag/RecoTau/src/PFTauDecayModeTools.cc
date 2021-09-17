#include <map>
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

namespace reco {
  namespace tau {

    namespace {
      struct entry {
        char const *label;
        reco::PFTau::hadronicDecayMode mode;
      };

      // Convert the string decay mode from PhysicsTools to the
      // PFTau::hadronicDecayMode format
      constexpr entry dmTranslatorMap[] = {{"unknown", reco::PFTau::kNull},
                                           {"rare", reco::PFTau::kRareDecayMode},
                                           {"oneProng0Pi0", reco::PFTau::kOneProng0PiZero},
                                           {"oneProng1Pi0", reco::PFTau::kOneProng1PiZero},
                                           {"oneProng2Pi0", reco::PFTau::kOneProng2PiZero},
                                           {"oneProngOther", reco::PFTau::kOneProngNPiZero},
                                           {"threeProng0Pi0", reco::PFTau::kThreeProng0PiZero},
                                           {"threeProng1Pi0", reco::PFTau::kThreeProng1PiZero},
                                           {"threeProngOther", reco::PFTau::kThreeProngNPiZero},
                                           {"electron", reco::PFTau::kNull},
                                           {"muon", reco::PFTau::kNull},
                                           {nullptr, reco::PFTau::kNull}};

      constexpr bool same(char const *x, char const *y) { return !*x && !*y ? true : (*x == *y && same(x + 1, y + 1)); }

      constexpr enum reco::PFTau::hadronicDecayMode decayModeStringToId(char const *label, entry const *entries) {
        return !entries->label               ? reco::PFTau::kRareDecayMode
               : same(entries->label, label) ? entries->mode
                                             : decayModeStringToId(label, entries + 1);
      }

      constexpr char const *decayModeIdToString(reco::PFTau::hadronicDecayMode mode, entry const *entries) {
        return !entries->label ? "unknown"
                               : (entries->mode == mode ? entries->label : decayModeIdToString(mode, entries + 1));
      }
    }  // namespace

    unsigned int chargedHadronsInDecayMode(PFTau::hadronicDecayMode mode) {
      int modeAsInt = static_cast<int>(mode);
      return (modeAsInt / PFTau::kOneProngNPiZero) + 1;
    }

    unsigned int piZerosInDecayMode(PFTau::hadronicDecayMode mode) {
      int modeAsInt = static_cast<int>(mode);
      return (modeAsInt % PFTau::kOneProngNPiZero);
    }

    PFTau::hadronicDecayMode translateDecayMode(unsigned int nCharged, unsigned int nPiZeros) {
      // If no tracks exist, this is definitely not a tau!
      if (!nCharged)
        return PFTau::kNull;
      // Find the maximum number of PiZeros our parameterization can hold
      const unsigned int maxPiZeros = PFTau::kOneProngNPiZero;
      // Determine our track index
      unsigned int trackIndex = (nCharged - 1) * (maxPiZeros + 1);
      // Check if we handle the given number of tracks
      if (trackIndex >= PFTau::kRareDecayMode)
        return PFTau::kRareDecayMode;

      nPiZeros = (nPiZeros <= maxPiZeros) ? nPiZeros : maxPiZeros;
      return static_cast<PFTau::hadronicDecayMode>(trackIndex + nPiZeros);
    }

    PFTau::hadronicDecayMode translateGenDecayModeToReco(const std::string &name) {
      return decayModeStringToId(name.c_str(), dmTranslatorMap);
    }

    std::string translateRecoDecayModeToGen(PFTau::hadronicDecayMode decayMode) {
      return decayModeIdToString(decayMode, dmTranslatorMap);
    }

    PFTau::hadronicDecayMode getDecayMode(const reco::GenJet *genJet) {
      if (!genJet)
        return reco::PFTau::kNull;
      return translateGenDecayModeToReco(JetMCTagUtils::genTauDecayMode(*genJet));
    }

  }  // namespace tau
}  // namespace reco

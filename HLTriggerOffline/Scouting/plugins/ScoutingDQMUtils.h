#ifndef HLTriggerOffline_Scouting_ScoutingDQMUtils_h
#define HLTriggerOffline_Scouting_ScoutingDQMUtils_h

#include <cmath>

#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "FWCore/Common/interface/TriggerNames.h"

namespace scoutingDQMUtils {

  // Constants
  static constexpr double ELECTRON_MASS = 0.000511;  // Electron mass in GeV
  static constexpr double ELE_etaEB = 1.479;         // Eta restriction to barrel (for electrons)

  // trigs
  inline double computePtFromEnergyMassEta(double energy, double mass, double eta) {
    double transverseEnergy = std::sqrt(energy * energy - mass * mass);
    return transverseEnergy / std::cosh(eta);
  }

  // scouting electron IB
  inline const bool scoutingElectronID(const Run3ScoutingElectron& el) {
    bool isEB = (std::abs(el.eta()) < scoutingDQMUtils::ELE_etaEB);
    if (isEB) {
      if (el.sigmaIetaIeta() > 0.015)
        return false;
      if (el.hOverE() > 0.2)
        return false;
      if (std::abs(el.dEtaIn()) > 0.008)
        return false;
      if (std::abs(el.dPhiIn()) > 0.06)
        return false;
      if (el.ecalIso() / el.pt() > 0.25)
        return false;
      return true;

    } else {
      if (el.sigmaIetaIeta() > 0.045)
        return false;
      if (el.hOverE() > 0.2)
        return false;
      if (std::abs(el.dEtaIn()) > 0.012)
        return false;
      if (std::abs(el.dPhiIn()) > 0.06)
        return false;
      if (el.ecalIso() / el.pt() > 0.1)
        return false;
      return true;
    }
  }

  inline bool scoutingElectronGsfTrackID(const Run3ScoutingElectron& el, size_t trackIdx) {
    if (trackIdx > el.trkpt().size())
      edm::LogError("ScoutingDQMUtils") << "Invalid track index for electron: Exceeds the number of tracks";

    math::PtEtaPhiMLorentzVector particleSC(el.pt(), el.eta(), el.phi(), scoutingDQMUtils::ELECTRON_MASS);
    math::PtEtaPhiMLorentzVector particleTrk(
        el.trkpt()[trackIdx], el.trketa()[trackIdx], el.trkphi()[trackIdx], scoutingDQMUtils::ELECTRON_MASS);

    double scEnergy = particleSC.energy();
    double trkEnergy = particleTrk.energy();
    double relEnergyDiff = std::abs(scEnergy - trkEnergy) / scEnergy;
    double dPhi = deltaPhi(particleSC.phi(), particleTrk.phi());

    bool isEB = (std::abs(el.eta()) < scoutingDQMUtils::ELE_etaEB);
    if (isEB) {
      if (el.trkpt()[trackIdx] < 12)
        return false;
      if (relEnergyDiff > 1)
        return false;
      if (dPhi > 0.06)
        return false;
      if (el.trkchi2overndf()[trackIdx] > 3)
        return false;
      return true;
    } else {
      if (el.trkpt()[trackIdx] < 12)
        return false;
      if (relEnergyDiff > 1)
        return false;
      if (dPhi > 0.06)
        return false;
      if (el.trkchi2overndf()[trackIdx] > 2)
        return false;
      return true;
    }
  }

  inline bool scoutingElectronGsfTrackIdx(const Run3ScoutingElectron& el, size_t& trackIdx) {
    bool foundGoodGsfTrkIdx = false;
    for (size_t i = 0; i < el.trkpt().size(); ++i) {
      if (scoutingDQMUtils::scoutingElectronGsfTrackID(el, i)) {
        if (!foundGoodGsfTrkIdx) {
          foundGoodGsfTrkIdx = true;
          trackIdx = i;
        } else {
          double relPtDiff = fabs(el.trkpt()[i] - el.pt()) / el.pt();
          double relPtDiffOld = fabs(el.trkpt()[trackIdx] - el.pt()) / el.pt();
          if (relPtDiff < relPtDiffOld)
            trackIdx = i;
        }
      }
    }
    return foundGoodGsfTrkIdx;
  }

  inline bool hasPatternInHLTPath(const edm::TriggerNames& triggerNames, const std::string& pattern) {
    for (unsigned int i = 0; i < triggerNames.size(); ++i) {
      const std::string& triggerName = triggerNames.triggerName(i);

      // Check if triggerName starts with the specified prefix
      if (triggerName.find(pattern) == 0) {  // Position 0 means it starts with 'prefix'
        return true;                         // Pattern match found
      }
    }
    return false;  // No match found
  }
}  // namespace scoutingDQMUtils

#endif  //

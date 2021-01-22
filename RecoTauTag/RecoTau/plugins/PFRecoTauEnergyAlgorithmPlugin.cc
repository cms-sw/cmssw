/*
 * =============================================================================
 *       Filename:  RecoTauEnergyAlgorithmPlugin.cc
 *
 *    Description:  Determine best estimate for tau energy
 *                  for tau candidates reconstructed in different decay modes
 *
 *        Created:  04/09/2013 11:40:00
 *
 *         Authors:  Christian Veelken (LLR)
 *
 * =============================================================================
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"
#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <vector>
#include <cmath>

namespace reco {
  namespace tau {

    class PFRecoTauEnergyAlgorithmPlugin : public RecoTauModifierPlugin {
    public:
      explicit PFRecoTauEnergyAlgorithmPlugin(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
      ~PFRecoTauEnergyAlgorithmPlugin() override;
      void operator()(reco::PFTau&) const override;
      void beginEvent() override;
      void endEvent() override;

    private:
      double dRaddNeutralHadron_;
      double minNeutralHadronEt_;
      double dRaddPhoton_;
      double minGammaEt_;

      int verbosity_;
    };

    PFRecoTauEnergyAlgorithmPlugin::PFRecoTauEnergyAlgorithmPlugin(const edm::ParameterSet& cfg,
                                                                   edm::ConsumesCollector&& iC)
        : RecoTauModifierPlugin(cfg, std::move(iC)),
          dRaddNeutralHadron_(cfg.getParameter<double>("dRaddNeutralHadron")),
          minNeutralHadronEt_(cfg.getParameter<double>("minNeutralHadronEt")),
          dRaddPhoton_(cfg.getParameter<double>("dRaddPhoton")),
          minGammaEt_(cfg.getParameter<double>("minGammaEt")),
          verbosity_(cfg.getParameter<int>("verbosity")) {}

    PFRecoTauEnergyAlgorithmPlugin::~PFRecoTauEnergyAlgorithmPlugin() {}

    void PFRecoTauEnergyAlgorithmPlugin::beginEvent() {}

    namespace {
      double getTrackPerr2(const reco::Track& track) {
        double trackPerr = track.p() * (track.ptError() / track.pt());
        return trackPerr * trackPerr;
      }

      void updateTauP4(reco::PFTau& tau, double sf, const reco::Candidate::LorentzVector& addP4) {
        // preserve tau candidate mass when adding extra neutral energy
        double tauPx_modified = tau.px() + sf * addP4.px();
        double tauPy_modified = tau.py() + sf * addP4.py();
        double tauPz_modified = tau.pz() + sf * addP4.pz();
        double tauMass = tau.mass();
        double tauEn_modified = sqrt(tauPx_modified * tauPx_modified + tauPy_modified * tauPy_modified +
                                     tauPz_modified * tauPz_modified + tauMass * tauMass);
        reco::Candidate::LorentzVector tauP4_modified(tauPx_modified, tauPy_modified, tauPz_modified, tauEn_modified);
        tau.setP4(tauP4_modified);
      }

      void killTau(reco::PFTau& tau) {
        reco::Candidate::LorentzVector tauP4_modified(0., 0., 0., 0.);
        tau.setP4(tauP4_modified);
        tau.setStatus(-1);
      }
    }  // namespace

    template <class Base, class Der>
    bool isPtrEqual(const edm::Ptr<Base>& b, const edm::Ptr<Der>& d) {
      return edm::Ptr<Der>(b) == d;
    }

    template <class Base>
    bool isPtrEqual(const edm::Ptr<Base>& b, const edm::Ptr<Base>& d) {
      return b == d;
    }

    void PFRecoTauEnergyAlgorithmPlugin::operator()(reco::PFTau& tau) const {
      if (verbosity_) {
        std::cout << "<PFRecoTauEnergyAlgorithmPlugin::operator()>:" << std::endl;
        std::cout << "tau: Pt = " << tau.pt() << ", eta = " << tau.eta() << ", phi = " << tau.phi()
                  << " (En = " << tau.energy() << ", decayMode = " << tau.decayMode() << ")" << std::endl;
      }

      // Add high Pt PFNeutralHadrons and PFGammas that are not "used" by tau decay mode object
      std::vector<reco::CandidatePtr> addNeutrals;
      reco::Candidate::LorentzVector addNeutralsSumP4;
      const auto& jetConstituents = tau.jetRef()->daughterPtrVector();
      for (const auto& jetConstituent : jetConstituents) {
        int jetConstituentPdgId = std::abs(jetConstituent->pdgId());
        if (!((jetConstituentPdgId == 130 && jetConstituent->et() > minNeutralHadronEt_) ||
              (jetConstituentPdgId == 22 && jetConstituent->et() > minGammaEt_)))
          continue;

        bool isSignalCand = false;
        const auto& signalCands = tau.signalCands();
        for (const auto& signalCand : signalCands) {
          if (isPtrEqual(jetConstituent, signalCand))
            isSignalCand = true;
        }
        if (isSignalCand)
          continue;

        double dR = deltaR(jetConstituent->p4(), tau.p4());
        double dRadd = -1.;
        if (jetConstituentPdgId == 130)
          dRadd = dRaddNeutralHadron_;
        else if (jetConstituentPdgId == 22)
          dRadd = dRaddPhoton_;
        if (dR < dRadd) {
          addNeutrals.push_back(jetConstituent);
          addNeutralsSumP4 += jetConstituent->p4();
        }
      }
      if (verbosity_) {
        std::cout << "addNeutralsSumP4: En = " << addNeutralsSumP4.energy() << std::endl;
      }

      unsigned numNonPFCandTracks = 0;
      double nonPFCandTracksSumP = 0.;
      double nonPFCandTracksSumPerr2 = 0.;
      const std::vector<PFRecoTauChargedHadron>& chargedHadrons = tau.signalTauChargedHadronCandidatesRestricted();
      for (std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
           chargedHadron != chargedHadrons.end();
           ++chargedHadron) {
        if (chargedHadron->algoIs(PFRecoTauChargedHadron::kTrack)) {
          ++numNonPFCandTracks;
          const reco::Track* chargedHadronTrack = getTrackFromChargedHadron(*chargedHadron);
          if (chargedHadronTrack != nullptr) {
            nonPFCandTracksSumP += chargedHadronTrack->p();
            nonPFCandTracksSumPerr2 += getTrackPerr2(*chargedHadronTrack);
          }
        }
      }
      if (verbosity_) {
        std::cout << "nonPFCandTracksSumP = " << nonPFCandTracksSumP << " +/- " << sqrt(nonPFCandTracksSumPerr2)
                  << " (numNonPFCandTracks = " << numNonPFCandTracks << ")" << std::endl;
      }

      if (numNonPFCandTracks == 0) {
        // This is the easy case:
        // All tau energy is taken from PFCandidates reconstructed by PFlow algorithm
        // and there is no issue with double-counting of energy.
        if (verbosity_) {
          std::cout << "easy case: all tracks are associated to PFCandidates --> leaving tau momentum untouched."
                    << std::endl;
        }
        updateTauP4(tau, 1., addNeutralsSumP4);
        return;
      } else {
        // This is the difficult case:
        // The tau energy needs to be computed for an arbitrary mix of charged and neutral PFCandidates plus reco::Tracks.
        // We need to make sure not to double-count energy deposited by reco::Track in ECAL and/or HCAL as neutral PFCandidates.

        // Check if we have enough energy in collection of PFNeutralHadrons and PFGammas that are not "used" by tau decay mode object
        // to balance track momenta:
        if (nonPFCandTracksSumP < addNeutralsSumP4.energy()) {
          double scaleFactor = 1. - nonPFCandTracksSumP / addNeutralsSumP4.energy();
          if (!(scaleFactor >= 0. && scaleFactor <= 1.)) {
            edm::LogWarning("PFRecoTauEnergyAlgorithmPlugin::operator()")
                << "Failed to compute tau energy --> killing tau candidate !!" << std::endl;
            killTau(tau);
            return;
          }
          if (verbosity_) {
            std::cout << "case (2): addNeutralsSumEn > nonPFCandTracksSumP --> adjusting tau momentum." << std::endl;
          }
          updateTauP4(tau, scaleFactor, addNeutralsSumP4);
          return;
        }

        // Determine which neutral PFCandidates are close to PFChargedHadrons
        // and have been merged into ChargedHadrons
        std::vector<reco::CandidatePtr> mergedNeutrals;
        reco::Candidate::LorentzVector mergedNeutralsSumP4;
        for (std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
             chargedHadron != chargedHadrons.end();
             ++chargedHadron) {
          if (chargedHadron->algoIs(PFRecoTauChargedHadron::kTrack)) {
            const std::vector<reco::CandidatePtr>& neutralPFCands = chargedHadron->getNeutralPFCandidates();
            for (std::vector<reco::CandidatePtr>::const_iterator neutralPFCand = neutralPFCands.begin();
                 neutralPFCand != neutralPFCands.end();
                 ++neutralPFCand) {
              mergedNeutrals.push_back(*neutralPFCand);
              mergedNeutralsSumP4 += (*neutralPFCand)->p4();
            }
          }
        }
        if (verbosity_) {
          std::cout << "mergedNeutralsSumP4: En = " << mergedNeutralsSumP4.energy() << std::endl;
        }

        // Check if track momenta are balanced by sum of PFNeutralHadrons and PFGammas that are not "used" by tau decay mode object
        // plus neutral PFCandidates close to PFChargedHadrons:
        if (nonPFCandTracksSumP < (addNeutralsSumP4.energy() + mergedNeutralsSumP4.energy())) {
          double scaleFactor = ((addNeutralsSumP4.energy() + mergedNeutralsSumP4.energy()) - nonPFCandTracksSumP) /
                               mergedNeutralsSumP4.energy();
          if (!(scaleFactor >= 0. && scaleFactor <= 1.)) {
            edm::LogWarning("PFRecoTauEnergyAlgorithmPlugin::operator()")
                << "Failed to compute tau energy --> killing tau candidate !!" << std::endl;
            killTau(tau);
            return;
          }
          reco::Candidate::LorentzVector diffP4;
          size_t numChargedHadrons = chargedHadrons.size();
          for (size_t iChargedHadron = 0; iChargedHadron < numChargedHadrons; ++iChargedHadron) {
            const PFRecoTauChargedHadron& chargedHadron = chargedHadrons[iChargedHadron];
            if (!chargedHadron.getNeutralPFCandidates().empty()) {
              PFRecoTauChargedHadron chargedHadron_modified = chargedHadron;
              setChargedHadronP4(chargedHadron_modified, scaleFactor);
              tau.signalTauChargedHadronCandidatesRestricted()[iChargedHadron] = chargedHadron_modified;
              diffP4 += (chargedHadron.p4() - chargedHadron_modified.p4());
            }
          }
          if (verbosity_) {
            std::cout << "case (3): (addNeutralsSumEn + mergedNeutralsSumEn) > nonPFCandTracksSumP --> adjusting tau "
                         "momentum."
                      << std::endl;
          }
          updateTauP4(tau, -1., diffP4);
          return;
        }

        // Determine energy sum of all PFNeutralHadrons interpreted as ChargedHadrons with missing track
        unsigned numChargedHadronNeutrals = 0;
        std::vector<reco::CandidatePtr> chargedHadronNeutrals;
        reco::Candidate::LorentzVector chargedHadronNeutralsSumP4;
        for (std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
             chargedHadron != chargedHadrons.end();
             ++chargedHadron) {
          if (chargedHadron->algoIs(PFRecoTauChargedHadron::kPFNeutralHadron)) {
            ++numChargedHadronNeutrals;
            chargedHadronNeutrals.push_back(chargedHadron->getChargedPFCandidate());
            chargedHadronNeutralsSumP4 += chargedHadron->getChargedPFCandidate()->p4();
          }
        }
        if (verbosity_) {
          std::cout << "chargedHadronNeutralsSumP4: En = " << chargedHadronNeutralsSumP4.energy()
                    << " (numChargedHadronNeutrals = " << numChargedHadronNeutrals << ")" << std::endl;
        }

        // Check if sum of PFNeutralHadrons and PFGammas that are not "used" by tau decay mode object
        // plus neutral PFCandidates close to PFChargedHadrons plus PFNeutralHadrons interpreted as ChargedHadrons with missing track balances track momenta
        if (nonPFCandTracksSumP <
            (addNeutralsSumP4.energy() + mergedNeutralsSumP4.energy() + chargedHadronNeutralsSumP4.energy())) {
          double scaleFactor =
              ((addNeutralsSumP4.energy() + mergedNeutralsSumP4.energy() + chargedHadronNeutralsSumP4.energy()) -
               nonPFCandTracksSumP) /
              chargedHadronNeutralsSumP4.energy();
          if (!(scaleFactor >= 0. && scaleFactor <= 1.)) {
            edm::LogWarning("PFRecoTauEnergyAlgorithmPlugin::operator()")
                << "Failed to compute tau energy --> killing tau candidate !!" << std::endl;
            killTau(tau);
            return;
          }
          reco::Candidate::LorentzVector diffP4;
          size_t numChargedHadrons = chargedHadrons.size();
          for (size_t iChargedHadron = 0; iChargedHadron < numChargedHadrons; ++iChargedHadron) {
            const PFRecoTauChargedHadron& chargedHadron = chargedHadrons[iChargedHadron];
            if (chargedHadron.algoIs(PFRecoTauChargedHadron::kPFNeutralHadron)) {
              PFRecoTauChargedHadron chargedHadron_modified = chargedHadron;
              chargedHadron_modified.neutralPFCandidates_.clear();
              const CandidatePtr& chargedPFCand = chargedHadron.getChargedPFCandidate();
              double chargedHadronPx_modified = scaleFactor * chargedPFCand->px();
              double chargedHadronPy_modified = scaleFactor * chargedPFCand->py();
              double chargedHadronPz_modified = scaleFactor * chargedPFCand->pz();
              reco::Candidate::LorentzVector chargedHadronP4_modified = compChargedHadronP4fromPxPyPz(
                  chargedHadronPx_modified, chargedHadronPy_modified, chargedHadronPz_modified);
              chargedHadron_modified.setP4(chargedHadronP4_modified);
              tau.signalTauChargedHadronCandidatesRestricted()[iChargedHadron] = chargedHadron_modified;
              diffP4 += (chargedHadron.p4() - chargedHadron_modified.p4());
            }
          }
          if (verbosity_) {
            std::cout << "case (4): (addNeutralsSumEn + mergedNeutralsSumEn + chargedHadronNeutralsSumEn) > "
                         "nonPFCandTracksSumP --> adjusting momenta of tau and chargedHadrons."
                      << std::endl;
          }
          updateTauP4(tau, -1., diffP4);
          return;
        } else {
          double allTracksSumP = 0.;
          double allTracksSumPerr2 = 0.;
          const std::vector<PFRecoTauChargedHadron> chargedHadrons = tau.signalTauChargedHadronCandidatesRestricted();
          for (std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
               chargedHadron != chargedHadrons.end();
               ++chargedHadron) {
            if (chargedHadron->algoIs(PFRecoTauChargedHadron::kChargedPFCandidate) ||
                chargedHadron->algoIs(PFRecoTauChargedHadron::kTrack)) {
              const reco::Track* chargedHadronTrack = getTrackFromChargedHadron(*chargedHadron);
              if (chargedHadronTrack != nullptr) {
                allTracksSumP += chargedHadronTrack->p();
                allTracksSumPerr2 += getTrackPerr2(*chargedHadronTrack);
              } else {
                // don't print warning as it is caused by missing detailed track inforamtion in lostTrack in miniAOD
                if (!(chargedHadron->algoIs(PFRecoTauChargedHadron::kTrack) &&
                      chargedHadron->getLostTrackCandidate().isNonnull() &&
                      chargedHadron->getLostTrackCandidate()->bestTrack() == nullptr)) {
                  edm::LogInfo("PFRecoTauEnergyAlgorithmPlugin::operator()")
                      << "PFRecoTauChargedHadron has no associated reco::Track !!";
                }
                if (verbosity_) {
                  chargedHadron->print();
                }
              }
            }
          }
          if (verbosity_) {
            std::cout << "allTracksSumP = " << allTracksSumP << " +/- " << sqrt(allTracksSumPerr2) << std::endl;
          }
          double allNeutralsSumEn = 0.;
          const auto& signalCands = tau.signalCands();
          for (const auto& signalCand : signalCands) {
            if (verbosity_) {
              std::cout << "Candidate #" << signalCand.id() << ":" << signalCand.key() << ":"
                        << " Pt = " << (signalCand)->pt() << ", eta = " << (signalCand)->eta()
                        << ", phi = " << (signalCand)->phi() << std::endl;
            }
            const PFCandidate* pfCand = dynamic_cast<const PFCandidate*>(&*signalCand);
            if (pfCand) {
              if (verbosity_) {
                std::cout << "calorimeter energy:"
                          << " ECAL = " << (pfCand)->ecalEnergy() << ","
                          << " HCAL = " << (pfCand)->hcalEnergy() << ","
                          << " HO = " << (pfCand)->hoEnergy() << std::endl;
              }
              if (edm::isFinite(pfCand->ecalEnergy()))
                allNeutralsSumEn += pfCand->ecalEnergy();
              if (edm::isFinite(pfCand->hcalEnergy()))
                allNeutralsSumEn += pfCand->hcalEnergy();
              if (edm::isFinite(pfCand->hoEnergy()))
                allNeutralsSumEn += pfCand->hoEnergy();
            } else {
              // TauReco@MiniAOD: individual ECAL and HCAL energies recovered from fractions. Info on HO energy is not (yet?) available in miniAOD.
              const pat::PackedCandidate* packedCand = dynamic_cast<const pat::PackedCandidate*>(&*signalCand);
              assert(packedCand);  // Taus are built either from reco::PFCandidates or pat::PackedCandidates
              double caloEn = packedCand->caloFraction() * packedCand->energy();
              double hcalEn = caloEn * packedCand->hcalFraction();
              double ecalEn = caloEn - hcalEn;
              if (verbosity_) {
                std::cout << "calorimeter energy:"
                          << " ECAL = " << ecalEn << ","
                          << " HCAL = " << hcalEn << ","
                          << " HO unknown for PackedCand" << std::endl;
              }
              if (edm::isFinite(ecalEn))
                allNeutralsSumEn += ecalEn;
              if (edm::isFinite(hcalEn))
                allNeutralsSumEn += hcalEn;
            }
          }
          allNeutralsSumEn += addNeutralsSumP4.energy();
          if (allNeutralsSumEn < 0.)
            allNeutralsSumEn = 0.;
          if (verbosity_) {
            std::cout << "allNeutralsSumEn = " << allNeutralsSumEn << std::endl;
          }
          if (allNeutralsSumEn > allTracksSumP) {
            // Adjust momenta of neutral PFCandidates merged into ChargedHadrons
            size_t numChargedHadrons = chargedHadrons.size();
            for (size_t iChargedHadron = 0; iChargedHadron < numChargedHadrons; ++iChargedHadron) {
              const PFRecoTauChargedHadron& chargedHadron = chargedHadrons[iChargedHadron];
              if (chargedHadron.algoIs(PFRecoTauChargedHadron::kChargedPFCandidate)) {
                PFRecoTauChargedHadron chargedHadron_modified = chargedHadron;
                chargedHadron_modified.neutralPFCandidates_.clear();
                chargedHadron_modified.setP4(chargedHadron.getChargedPFCandidate()->p4());
                if (verbosity_) {
                  std::cout << "chargedHadron #" << iChargedHadron << ": changing En = " << chargedHadron.energy()
                            << " to " << chargedHadron_modified.energy() << std::endl;
                }
                tau.signalTauChargedHadronCandidatesRestricted()[iChargedHadron] = chargedHadron_modified;
              } else if (chargedHadron.algoIs(PFRecoTauChargedHadron::kTrack)) {
                PFRecoTauChargedHadron chargedHadron_modified = chargedHadron;
                chargedHadron_modified.neutralPFCandidates_.clear();
                reco::Candidate::LorentzVector chargedHadronP4_modified(0., 0., 0., 0.);
                const reco::Track* chTrack = getTrackFromChargedHadron(chargedHadron);
                if (chTrack != nullptr) {
                  double chargedHadronPx_modified = chTrack->px();
                  double chargedHadronPy_modified = chTrack->py();
                  double chargedHadronPz_modified = chTrack->pz();
                  chargedHadronP4_modified = compChargedHadronP4fromPxPyPz(
                      chargedHadronPx_modified, chargedHadronPy_modified, chargedHadronPz_modified);
                } else {
                  // don't print warning as it is caused by missing detailed track inforamtion in lostTrack in miniAOD
                  if (!(chargedHadron.algoIs(PFRecoTauChargedHadron::kTrack) &&
                        chargedHadron.getLostTrackCandidate().isNonnull() &&
                        chargedHadron.getLostTrackCandidate()->bestTrack() == nullptr)) {
                    edm::LogWarning("PFRecoTauEnergyAlgorithmPlugin::operator()")
                        << "PFRecoTauChargedHadron has no associated reco::Track !!" << std::endl;
                  }
                  if (verbosity_) {
                    chargedHadron.print();
                  }
                }
                chargedHadron_modified.setP4(chargedHadronP4_modified);
                if (verbosity_) {
                  std::cout << "chargedHadron #" << iChargedHadron << ": changing En = " << chargedHadron.energy()
                            << " to " << chargedHadron_modified.energy() << std::endl;
                }
                tau.signalTauChargedHadronCandidatesRestricted()[iChargedHadron] = chargedHadron_modified;
              }
            }
            double scaleFactor = allNeutralsSumEn / tau.energy();
            if (verbosity_) {
              std::cout << "case (5): allNeutralsSumEn > allTracksSumP --> adjusting momenta of tau and chargedHadrons."
                        << std::endl;
            }
            updateTauP4(tau, scaleFactor - 1., tau.p4());
            return;
          } else {
            if (numChargedHadronNeutrals == 0 && tau.signalPiZeroCandidates().empty()) {
              // Adjust momenta of ChargedHadrons build from reco::Tracks to match sum of energy deposits in ECAL + HCAL + HO
              size_t numChargedHadrons = chargedHadrons.size();
              for (size_t iChargedHadron = 0; iChargedHadron < numChargedHadrons; ++iChargedHadron) {
                const PFRecoTauChargedHadron& chargedHadron = chargedHadrons[iChargedHadron];
                if (chargedHadron.algoIs(PFRecoTauChargedHadron::kChargedPFCandidate) ||
                    chargedHadron.algoIs(PFRecoTauChargedHadron::kTrack)) {
                  PFRecoTauChargedHadron chargedHadron_modified = chargedHadron;
                  chargedHadron_modified.neutralPFCandidates_.clear();
                  reco::Candidate::LorentzVector chargedHadronP4_modified(0., 0., 0., 0.);
                  const reco::Track* chargedHadronTrack = getTrackFromChargedHadron(chargedHadron);
                  if (chargedHadronTrack != nullptr) {
                    double trackP = chargedHadronTrack->p();
                    double trackPerr2 = getTrackPerr2(*chargedHadronTrack);
                    if (verbosity_) {
                      std::cout << "trackP = " << trackP << " +/- " << sqrt(trackPerr2) << std::endl;
                    }
                    // CV: adjust track momenta such that difference beeen (measuredTrackP - adjustedTrackP)/sigmaMeasuredTrackP is minimal
                    //    (expression derived using Mathematica)
                    double trackP_modified = (trackP * (allTracksSumPerr2 - trackPerr2) +
                                              trackPerr2 * (allNeutralsSumEn - (allTracksSumP - trackP))) /
                                             allTracksSumPerr2;
                    // CV: trackP_modified may actually become negative in case sum of energy deposits in ECAL + HCAL + HO is small
                    //     and one of the tracks has a significantly larger momentum uncertainty than the other tracks.
                    //     In this case set track momentum to small positive value.
                    if (trackP_modified < 1.e-1)
                      trackP_modified = 1.e-1;
                    if (verbosity_) {
                      std::cout << "trackP (modified) = " << trackP_modified << std::endl;
                    }
                    double scaleFactor = trackP_modified / trackP;
                    if (!(scaleFactor >= 0. && scaleFactor <= 1.)) {
                      edm::LogWarning("PFRecoTauEnergyAlgorithmPlugin::operator()")
                          << "Failed to compute tau energy --> killing tau candidate !!" << std::endl;
                      killTau(tau);
                      return;
                    }
                    double chargedHadronPx_modified = scaleFactor * chargedHadronTrack->px();
                    double chargedHadronPy_modified = scaleFactor * chargedHadronTrack->py();
                    double chargedHadronPz_modified = scaleFactor * chargedHadronTrack->pz();
                    chargedHadronP4_modified = compChargedHadronP4fromPxPyPz(
                        chargedHadronPx_modified, chargedHadronPy_modified, chargedHadronPz_modified);
                  } else {
                    // don't print warning as it is caused by missing detailed track inforamtion in lostTrack in miniAOD
                    if (!(chargedHadron.algoIs(PFRecoTauChargedHadron::kTrack) &&
                          chargedHadron.getLostTrackCandidate().isNonnull() &&
                          chargedHadron.getLostTrackCandidate()->bestTrack() == nullptr)) {
                      edm::LogInfo("PFRecoTauEnergyAlgorithmPlugin::operator()")
                          << "PFRecoTauChargedHadron has no associated reco::Track !!";
                    }
                    if (verbosity_) {
                      chargedHadron.print();
                    }
                  }
                  chargedHadron_modified.setP4(chargedHadronP4_modified);
                  if (verbosity_) {
                    std::cout << "chargedHadron #" << iChargedHadron << ": changing En = " << chargedHadron.energy()
                              << " to " << chargedHadron_modified.energy() << std::endl;
                  }
                  tau.signalTauChargedHadronCandidatesRestricted()[iChargedHadron] = chargedHadron_modified;
                }
              }
              double scaleFactor = allNeutralsSumEn / tau.energy();
              if (verbosity_) {
                std::cout
                    << "case (6): allNeutralsSumEn < allTracksSumP --> adjusting momenta of tau and chargedHadrons."
                    << std::endl;
              }
              updateTauP4(tau, scaleFactor - 1., tau.p4());
              return;
            } else {
              // Interpretation of PFNeutralHadrons as ChargedHadrons with missing track and/or reconstruction of extra PiZeros
              // is not compatible with the fact that sum of reco::Track momenta exceeds sum of energy deposits in ECAL + HCAL + HO:
              // kill tau candidate (by setting its four-vector to zero)
              if (verbosity_) {
                std::cout << "case (7): allNeutralsSumEn < allTracksSumP not compatible with tau decay mode hypothesis "
                             "--> killing tau candidate."
                          << std::endl;
              }
              killTau(tau);
              return;
            }
          }
        }
      }

      // CV: You should never come here.
      if (verbosity_) {
        std::cout << "undefined case: you should never come here !!" << std::endl;
      }
      assert(0);
    }

    void PFRecoTauEnergyAlgorithmPlugin::endEvent() {}

  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
                  reco::tau::PFRecoTauEnergyAlgorithmPlugin,
                  "PFRecoTauEnergyAlgorithmPlugin");

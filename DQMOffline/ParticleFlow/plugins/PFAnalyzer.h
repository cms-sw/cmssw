#ifndef PFAnalyzer_H
#define PFAnalyzer_H

/** \class JetMETAnalyzer
 *
 *  DQM PF candidate analysis monitoring
 *
 *  \author J. Roloff - Brown University
 *
 */

#include <memory>
#include <fstream>
#include <utility>
#include <string>
#include <cmath>
#include <map>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PFParticle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
class PFAnalyzer;

class PFAnalyzer : public DQMEDAnalyzer {
public:
  /// Constructor
  PFAnalyzer(const edm::ParameterSet&);

  /// Destructor
  ~PFAnalyzer() override;

  /// Inizialize parameters for histo binning
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// Initialize run-based parameters
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct binInfo;
  // A map between an observable name and a function that obtains that observable from a  PFCandidate.
  // This allows us to construct more complicated observables easily, and have it more configurable
  // in the config file.
  std::map<std::string,
           std::function<double(const reco::PFCandidate, const pat::PackedCandidate, const reco::CandidatePtr, int)>>
      m_funcMap;
  //std::map<std::string, std::function<double(const reco::PFCandidateCollection, reco::PFCandidate::ParticleType pfType)>>
  std::map<std::string,
           std::function<double(const std::vector<reco::PFCandidate>, reco::PFCandidate::ParticleType pfType)>>
      m_eventFuncMap;

  std::map<std::string,
           std::function<double(
               const std::vector<reco::PFCandidatePtr> pfCands, reco::PFCandidate::ParticleType pfType, const reco::Jet)>>
      m_jetWideFuncMap;

  std::map<std::string, std::function<double(const reco::PFCandidate, const reco::Jet)>> m_pfInJetFuncMap;
  std::map<std::string, std::function<double(const reco::Jet, const std::vector<reco::PFCandidatePtr> pfCands)>>
      m_jetFuncMap;

  //std::map<std::string, std::function<bool(const edm::Handle<std::vector<reco::Jet> >& pfJets)>> m_eventSelectionMap;
  std::map<std::string, std::function<bool(const std::vector<reco::Jet>& pfJets)>> m_eventSelectionMap;

  binInfo getBinInfo(std::string);

  // Book MonitorElements
  void bookMESetSelection(std::string, DQMStore::IBooker&);

  int getPFBin(const reco::PFCandidate pfCand,
               const pat::PackedCandidate packedCand,
               const reco::CandidatePtr cand,
               int partType,
               int i);
  int getJetBin(const reco::Jet jetCand, const std::vector<reco::PFCandidatePtr> pfCands, int i);

  int getBinNumber(double binVal, std::vector<double> bins);
  int getBinNumbers(std::vector<double> binVal, std::vector<std::vector<double>> bins);
  std::vector<double> getBinList(std::string binString);

  std::vector<std::string> getAllSuffixes(std::vector<std::string> observables,
                                          std::vector<std::vector<double>> binnings);
  std::string stringWithDecimals(int bin, std::vector<double> bins);

  std::string getSuffix(std::vector<int> binList,
                        std::vector<std::string> observables,
                        std::vector<std::vector<double>> binnings);

  static double getEnergySpectrum(const reco::PFCandidate pfCand, const reco::Jet jet) {
    if (!jet.pt())
      return -1;
    return pfCand.pt() / jet.pt();
  }

  static double getNPFC(const std::vector<reco::PFCandidate> pfCands, reco::PFCandidate::ParticleType pfType) {
    int nPF = 0;
    for (const auto& pfCand : pfCands) {
      // We use X to indicate all
      if (pfCand.particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X) {
        nPF++;
      }
    }
    return nPF;
  }

  static double getNPFCinJet(const std::vector<reco::PFCandidatePtr> pfCands,
                             reco::PFCandidate::ParticleType pfType,
                             const reco::Jet jet) {
    int nPF = 0;
    for (const auto& pfCand : pfCands) {
      if (!pfCand)
        continue;
      // We use X to indicate all
      if (pfCand->particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X)
        nPF++;
    }
    return nPF;
  }

  static double getMaxPt(const std::vector<reco::PFCandidate> pfCands, reco::PFCandidate::ParticleType pfType) {
    double maxPt = 0;
    for (const auto& pfCand : pfCands) {
      // We use X to indicate all
      if (pfCand.particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X) {
        if (pfCand.pt() > maxPt)
          maxPt = pfCand.pt();
      }
    }
    return maxPt;
  }

  static double getMaxPtFracJet(const std::vector<reco::PFCandidatePtr> pfCands,
                                reco::PFCandidate::ParticleType pfType,
                                const reco::Jet jet) {
    double maxPt = 0;
    for (const auto& pfCand : pfCands) {
      if (!pfCand)
        continue;
      // We use X to indicate all
      if (pfCand->particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X)
        if (pfCand->pt() > maxPt)
          maxPt = pfCand->pt();
    }
    return maxPt / jet.pt();
  }

  // Various functions designed to get information from a PF canddidate
  static double getPt(const reco::PFCandidate pfCand,
                      const pat::PackedCandidate packedPart,
                      const reco::CandidatePtr cand,
                      int partType) {
    if (partType == 1) {
      return packedPart.pt();
    }
    if (partType == 2) {
      return cand->pt();
    }
    return pfCand.pt();
  }
  static double getEnergy(const reco::PFCandidate pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType) {
    if (partType == 1) {
      return packedPart.energy();
    }
    return pfCand.energy();
  }
  static double getEta(const reco::PFCandidate pfCand,
                       const pat::PackedCandidate packedPart,
                       const reco::CandidatePtr cand,
                       int partType) {
    if (partType == 1) {
      return packedPart.eta();
    }
    if (partType == 2) {
      return cand->eta();
    }
    return pfCand.eta();
  }
  static double getAbsEta(const reco::PFCandidate pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType) {
    if (partType == 1) {
      return std::abs(packedPart.eta());
    }
    if (partType == 2) {
      return std::abs(cand->eta());
    }
    return std::abs(pfCand.eta());
  }
  static double getPhi(const reco::PFCandidate pfCand,
                       const pat::PackedCandidate packedPart,
                       const reco::CandidatePtr cand,
                       int partType) {
    if (partType == 1) {
      return packedPart.phi();
    }
    if (partType == 2) {
      return cand->phi();
    }
    return pfCand.phi();
  }

  static double getHadCalibration(const reco::PFCandidate pfCand,
                                  const pat::PackedCandidate packedPart,
                                  const reco::CandidatePtr cand,
                                  int partType) {
    if (pfCand.rawHcalEnergy() == 0)
      return -1;
    return pfCand.hcalEnergy() / pfCand.rawHcalEnergy();
  }
  static double getPuppiWeight(const reco::PFCandidate pfCand,
                               const pat::PackedCandidate packedPart,
                               const reco::CandidatePtr cand,
                               int partType) {
    if (partType == 1) {
      return packedPart.puppiWeight();
    }
    return 1;
  }

  static double getTime(const reco::PFCandidate pfCand,
                        const pat::PackedCandidate packedPart,
                        const reco::CandidatePtr cand,
                        int partType) {
    if (partType == 1) {
      return packedPart.time();
    }
    return pfCand.time();
  }

  static double getHcalEnergy_depth1(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(1);
  }
  static double getHcalEnergy_depth2(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(2);
  }
  static double getHcalEnergy_depth3(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(3);
  }
  static double getHcalEnergy_depth4(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(4);
  }
  static double getHcalEnergy_depth5(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(5);
  }
  static double getHcalEnergy_depth6(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(6);
  }
  static double getHcalEnergy_depth7(const reco::PFCandidate pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType) {
    return pfCand.hcalDepthEnergyFraction(7);
  }

  static double getEcalEnergy(const reco::PFCandidate pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType) {
    if (partType == 1) {
      return (1.0 - packedPart.hcalFraction()) * packedPart.energy();
    }
    return pfCand.ecalEnergy();
  }
  static double getRawEcalEnergy(const reco::PFCandidate pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType) {
    if (partType == 1) {
      return (1.0 - packedPart.rawHcalFraction()) * packedPart.energy();
    }
    return pfCand.rawEcalEnergy();
  }
  static double getHcalEnergy(const reco::PFCandidate pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType) {
    if (partType == 1) {
      return packedPart.hcalFraction() * packedPart.energy();
    }
    return pfCand.hcalEnergy();
  }
  static double getRawHcalEnergy(const reco::PFCandidate pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType) {
    if (partType == 1) {
      return packedPart.rawHcalFraction() * packedPart.energy();
    }
    return pfCand.rawHcalEnergy();
  }
  static double getHOEnergy(const reco::PFCandidate pfCand,
                            const pat::PackedCandidate packedPart,
                            const reco::CandidatePtr cand,
                            int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.hoEnergy();
  }
  static double getRawHOEnergy(const reco::PFCandidate pfCand,
                               const pat::PackedCandidate packedPart,
                               const reco::CandidatePtr cand,
                               int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.rawHoEnergy();
  }

  static double getMVAIsolated(const reco::PFCandidate pfCand,
                               const pat::PackedCandidate packedPart,
                               const reco::CandidatePtr cand,
                               int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_Isolated();
  }
  static double getMVAEPi(const reco::PFCandidate pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_e_pi();
  }
  static double getMVAEMu(const reco::PFCandidate pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_e_mu();
  }
  static double getMVAPiMu(const reco::PFCandidate pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_pi_mu();
  }
  static double getMVANothingGamma(const reco::PFCandidate pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_nothing_gamma();
  }
  static double getMVANothingNH(const reco::PFCandidate pfCand,
                                const pat::PackedCandidate packedPart,
                                const reco::CandidatePtr cand,
                                int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_nothing_nh();
  }
  static double getMVAGammaNH(const reco::PFCandidate pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.mva_gamma_nh();
  }

  static double getDNNESigIsolated(const reco::PFCandidate pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.dnn_e_sigIsolated();
  }
  static double getDNNESigNonIsolated(const reco::PFCandidate pfCand,
                                      const pat::PackedCandidate packedPart,
                                      const reco::CandidatePtr cand,
                                      int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.dnn_e_sigNonIsolated();
  }
  static double getDNNEBkgNonIsolated(const reco::PFCandidate pfCand,
                                      const pat::PackedCandidate packedPart,
                                      const reco::CandidatePtr cand,
                                      int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.dnn_e_bkgNonIsolated();
  }
  static double getDNNEBkgTauIsolated(const reco::PFCandidate pfCand,
                                      const pat::PackedCandidate packedPart,
                                      const reco::CandidatePtr cand,
                                      int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.dnn_e_bkgTau();
  }
  static double getDNNEBkgPhotonIsolated(const reco::PFCandidate pfCand,
                                         const pat::PackedCandidate packedPart,
                                         const reco::CandidatePtr cand,
                                         int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.dnn_e_bkgPhoton();
  }

  static double getECalEFrac(const reco::PFCandidate pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.ecalEnergy() / pfCand.energy();
  }
  static double getHCalEFrac(const reco::PFCandidate pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.hcalEnergy() / pfCand.energy();
  }
  static double getPS1Energy(const reco::PFCandidate pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.pS1Energy();
  }
  static double getPS2Energy(const reco::PFCandidate pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.pS2Energy();
  }
  static double getPSEnergy(const reco::PFCandidate pfCand,
                            const pat::PackedCandidate packedPart,
                            const reco::CandidatePtr cand,
                            int partType) {
    if (partType) {
      return 0;
    }
    return pfCand.pS1Energy() + pfCand.pS2Energy();
  }

  static double getTrackPt(const reco::PFCandidate pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->pt();
    return 0;
  }

  static double getTrackNStripHits(const reco::PFCandidate pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->hitPattern().numberOfValidStripHits();
    return -1;
  }

  static double getTrackNPixHits(const reco::PFCandidate pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->hitPattern().numberOfValidPixelHits();

    return -1;
  }

  static double getTrackChi2(const reco::PFCandidate pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->chi2();
    return -1;
  }

  static double getTrackPtError(const reco::PFCandidate pfCand,
                                const pat::PackedCandidate packedPart,
                                const reco::CandidatePtr cand,
                                int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->ptError();
    return -1;
  }

  static double getTrackRelPtError(const reco::PFCandidate pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->ptError() / (pfCand.trackRef())->pt();
    return -1;
  }

  static double getTrackD0(const reco::PFCandidate pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->d0();
    return -1;
  }

  static double getTrackZ0(const reco::PFCandidate pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->d0();
    return -1;
  }

  static double getTrackThetaError(const reco::PFCandidate pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->thetaError();
    return -1;
  }

  static double getTrackEtaError(const reco::PFCandidate pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->etaError();
    return -1;
  }

  static double getTrackPhiError(const reco::PFCandidate pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType) {
    if (partType) {
      return 0;
    }
    if (pfCand.trackRef().isNonnull())
      return (pfCand.trackRef())->phiError();
    return -1;
  }

  static double getEoverP(const reco::PFCandidate pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType) {
    if (partType) {
      return 0;
    }
    double energy = 0;
    int maxElement = pfCand.elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL ||
              elements[iEle].type() == reco::PFBlockElement::ECAL) {  // Element is HB or HE
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            const reco::PFCluster& cluster = *clusterref;
            energy += cluster.energy();
          }
        }
      }
    }
    return energy / pfCand.p();
  }

  static double getHCalEnergy(const reco::PFCandidate pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType) {
    if (partType) {
      return 0;
    }
    double energy = 0;
    int maxElement = pfCand.elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL) {  // Element is HB or HE
            // Get cluster and hits
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            const reco::PFCluster& cluster = *clusterref;
            energy += cluster.energy();
          }
        }
      }
    }
    return energy;
  }

  static double getECalEnergy(const reco::PFCandidate pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType) {
    if (partType) {
      return 0;
    }
    double energy = 0;
    int maxElement = pfCand.elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::ECAL) {  // Element is HB or HE
            // Get cluster and hits
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            // When we don't have isolated tracks, this will be a bit useless, since the energy is shared across multiple tracks
            const reco::PFCluster& cluster = *clusterref;
            energy += cluster.energy();
          }
        }
      }
    }
    return energy;
  }

  static double getNTracksInBlock(const reco::PFCandidate pfCand,
                                  const pat::PackedCandidate packedPart,
                                  const reco::CandidatePtr cand,
                                  int partType) {
    if (partType) {
      return 0;
    }
    // We need this function to return a double, even though this is an integer value
    double nTrack = 0;
    int maxElement = pfCand.elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::TRACK) {  // Element is HB or HE
            nTrack += 1;
          }
        }
      }
    }
    return nTrack;
  }

  static double getJetPt(const reco::Jet jet, const std::vector<reco::PFCandidatePtr> pfCands) { return jet.pt(); }
  static double getJetChargeFrac(const reco::Jet jet, const std::vector<reco::PFCandidatePtr> pfCands) {
    double chargeFrac = 0;
    //std::vector<reco::PFCandidatePtr> pfConstits = jet.getPFConstituents();
    //std::vector<edm::Ptr<reco::Candidate>> pfConstits = jet.getJetConstituents();

    for (const auto& recoPF : pfCands) {
      if (recoPF->particleId() == reco::PFCandidate::ParticleType::h ||
          recoPF->particleId() == reco::PFCandidate::ParticleType::e ||
          recoPF->particleId() == reco::PFCandidate::ParticleType::mu)
        chargeFrac += recoPF->pt();
    }
    return chargeFrac / jet.pt();
  }

  bool passesTriggerSelection(const std::vector<reco::Jet>& pfJets,
                              const edm::Handle<edm::TriggerResults>& triggerResults,
                              const edm::TriggerNames& triggerNames,
                              const std::vector<std::string> triggerOptions) {
    // Hack to make it pass the lowest unprescaled HLT?
    Int_t JetHiPass = 0;

    const unsigned int nTrig(triggerNames.size());
    for (unsigned int i = 0; i < nTrig; ++i) {
      for (unsigned int j = 0; j < triggerOptions.size(); ++j) {
        if (triggerOptions[j].empty()) {
          JetHiPass = 1;
          break;
        }
        if (triggerNames.triggerName(i).find(triggerOptions[j]) != std::string::npos && triggerResults->accept(i)) {
          JetHiPass = 1;
          break;
        }
      }
      if (JetHiPass)
        break;
    }

    if (!JetHiPass)
      return false;
    return true;
  }

  static bool passesNoCutSelection(const std::vector<reco::Jet>& pfJets) { return true; }

  static bool passesDijetSelection(const std::vector<reco::Jet>& pfJets) {
    if (pfJets.size() < 2)
      return false;
    if (pfJets[0].pt() < 450)
      return false;
    if (pfJets[0].pt() / pfJets[1].pt() > 2)
      return false;

    return true;
  }

  static bool passesAnomalousSelection(const std::vector<reco::Jet>& pfJets) {
    if (pfJets.size() < 2)
      return false;
    if (pfJets[0].pt() < 450)
      return false;
    if (pfJets[1].pt() / pfJets[0].pt() > 0.5)
      return false;

    return true;
  }

  bool m_isMiniAOD;

  edm::EDGetTokenT<reco::PFCandidateCollection> thePfCandidateCollection_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> patPfCandidateCollection_;

  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<pat::JetCollection> patJetsToken_;

  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;
  edm::InputTag srcWeights;

  edm::EDGetTokenT<edm::ValueMap<float>> puppiWeightToken_;
  edm::Handle<edm::ValueMap<float>> puppiWeight;

  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;

  edm::InputTag theTriggerResultsLabel_;
  edm::InputTag vertexTag_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  std::string m_selection;

  std::vector<std::vector<std::string>> m_allSuffixes;
  std::vector<std::vector<std::string>> m_allJetSuffixes;

  // The directory where the output is stored
  std::string m_directory;

  // All of the histograms, stored as a map between the histogram name and the histogram
  std::map<std::string, MonitorElement*> map_of_MEs;

  std::map<reco::PFCandidate::ParticleType, std::string> m_particleTypeName;

  //check later if we need only one set of parameters
  edm::ParameterSet parameters_;

  typedef std::vector<std::string> vstring;
  typedef std::vector<double> vDouble;

  vstring m_triggerOptions;
  // Information on which observables to make histograms for.
  // In the config file, this should come as a comma-separated list of
  // the observable name, the number of bins for the histogram, and
  // the lowest and highest values for the histogram.
  // The observable name should have an entry in m_funcMap to define how
  // it can be retrieved from a PFCandidate.
  vstring m_observables;
  vstring m_eventObservables;
  vstring m_pfInJetObservables;

  vstring m_observableNames;
  vstring m_eventObservableNames;
  vstring m_pfInJetObservableNames;

  // Information on what cuts should be applied to PFCandidates that are
  // being monitored. In the config file, this should come as a comma-separated list of
  // the observable name, and the lowest and highest values for the histogram.
  // The observable name should have an entry in m_funcMap to define how
  // it can be retrieved from a PFCandidate.
  vstring m_cutList;
  std::vector<std::vector<std::string>> m_fullCutList;
  std::vector<std::vector<std::vector<double>>> m_binList;

  // Binning information for 2D histograms
  // Technically, these could be made using just the 1D cuts,
  // but this is useful for saving a bit of memory by creating fewer histograms.
  vstring m_cutList2D;
  std::vector<std::vector<std::string>> m_fullCutList2D;
  std::vector<std::vector<std::vector<double>>> m_binList2D;

  // Information on what cuts should be applied to PFJets, in the case that we
  // match PFCs to jets.In the config file, this should come as a comma-separated list of
  // the observable name, and the lowest and highest values for the histogram.
  // The observable name should have an entry in m_jetFuncMap to define how
  // it can be retrieved from a PFJet.
  vstring m_jetCutList;
  std::vector<std::vector<std::string>> m_fullJetCutList;
  std::vector<std::vector<std::vector<double>>> m_jetBinList;

  vDouble m_npvBins;

  // The dR radius used to match PFCs to jets.
  // Making this configurable is useful in case you want to look at the core of a jet.
  double m_matchingRadius;

  std::vector<std::string> m_pfNames;
};

struct PFAnalyzer::binInfo {
  std::string observable;
  std::string axisName;
  int nBins;
  double binMin;
  double binMax;
};

DEFINE_FWK_MODULE(PFAnalyzer);
#endif

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

#include "DataFormats/Common/interface/ValueMap.h"

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
           std::function<double(const reco::PFCandidatePtr, const pat::PackedCandidate, const reco::CandidatePtr, int, edm::Handle<edm::ValueMap<float>> )>>
      m_funcMap;
  std::map<std::string,
           std::function<double(const std::vector<reco::PFCandidatePtr>, reco::PFCandidate::ParticleType pfType)>>
      m_eventFuncMap;

  std::map<std::string,
           std::function<double(
               const std::vector<reco::PFCandidatePtr> pfCands, reco::PFCandidate::ParticleType pfType, const reco::Jet)>>
      m_jetWideFuncMap;

  std::map<std::string, std::function<double(const reco::PFCandidatePtr, const pat::PackedCandidate, const reco::CandidatePtr, int, const reco::Jet)>> m_pfInJetFuncMap;
  std::map<std::string, std::function<double(const reco::Jet, const std::vector<reco::PFCandidatePtr> pfCands)>>
      m_jetFuncMap;

  std::map<std::string, std::function<bool(const std::vector<reco::Jet>& pfJets)>> m_eventSelectionMap;

  binInfo getBinInfo(std::string);

  // Book MonitorElements
  void bookMESetSelection(std::string, DQMStore::IBooker&);

  int getPFBin(const reco::PFCandidatePtr pfCand,
               const pat::PackedCandidate packedCand,
               const reco::CandidatePtr cand,
               int partType,
               int i,
               edm::Handle<edm::ValueMap<float>> puppiWeight);
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



  static double getEnergySpectrum(const reco::PFCandidatePtr pfCand, const pat::PackedCandidate packedPart, const reco::CandidatePtr cand, int partType, const reco::Jet jet) {
    if (!jet.pt())
      return -1;
    if(partType==0){
      return pfCand.get()->pt() / jet.pt();
    }
    if(partType == 1){
      return packedPart.pt() / jet.pt();
    }
    if(partType == 2){
      return cand->pt() / jet.pt();
    }
    return -1;
  }

  static double getNPFC(const std::vector<reco::PFCandidatePtr> pfCands, reco::PFCandidate::ParticleType pfType) {
    int nPF = 0;
    for (const auto& pfCand : pfCands) {
      // We use X to indicate all
      if (pfCand.get()->particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X) {
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
      if (pfCand.get()->particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X)
        nPF++;
    }
    return nPF;
  }

  static double getMaxPt(const std::vector<reco::PFCandidatePtr> pfCands, reco::PFCandidate::ParticleType pfType) {
    double maxPt = 0;
    for (const auto& pfCand : pfCands) {
      // We use X to indicate all
      if (pfCand.get()->particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X) {
        if (pfCand.get()->pt() > maxPt)
          maxPt = pfCand.get()->pt();
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
      if (pfCand.get()->particleId() == pfType || pfType == reco::PFCandidate::ParticleType::X)
        if (pfCand.get()->pt() > maxPt)
          maxPt = pfCand.get()->pt();
    }
    return maxPt / jet.pt();
  }

  // Various functions designed to get information from a PF canddidate
  static double getPt(const reco::PFCandidatePtr pfCand,
                      const pat::PackedCandidate packedPart,
                      const reco::CandidatePtr cand,
                      int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->pt();
    }
    if (partType == 1) {
      return packedPart.pt();
    }
    if (partType == 2) {
      return cand->pt();
    }
    return -1;
  }

  static double getEnergy(const reco::PFCandidatePtr pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->energy();
    }
    if (partType == 1) {
      return packedPart.energy();
    }
    if (partType == 2){
      return cand->energy();
    }
    return -1;
  }
  static double getEta(const reco::PFCandidatePtr pfCand,
                       const pat::PackedCandidate packedPart,
                       const reco::CandidatePtr cand,
                       int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->eta();
    }
    if (partType == 1) {
      return packedPart.eta();
    }
    if (partType == 2) {
      return cand->eta();
    }
    return -1;
  }
  static double getAbsEta(const reco::PFCandidatePtr pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return std::abs(pfCand.get()->eta());
    }
    if (partType == 1) {
      return std::abs(packedPart.eta());
    }
    if (partType == 2) {
      return std::abs(cand->eta());
    }
    return -1;
  }
  static double getPhi(const reco::PFCandidatePtr pfCand,
                       const pat::PackedCandidate packedPart,
                       const reco::CandidatePtr cand,
                       int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->phi();
    }
    if (partType == 1) {
      return packedPart.phi();
    }
    if (partType == 2) {
      return cand->phi();
    }
    return -1;
  }

  static double getHadCalibration(const reco::PFCandidatePtr pfCand,
                                  const pat::PackedCandidate packedPart,
                                  const reco::CandidatePtr cand,
                                  int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0 ){
      if (pfCand.get()->rawHcalEnergy() == 0)
        return -1;
      return pfCand.get()->hcalEnergy() / pfCand.get()->rawHcalEnergy();
    }
    return -1;
  }
  static double getPuppiWeight(const reco::PFCandidatePtr pfCand,
                               const pat::PackedCandidate packedPart,
                               const reco::CandidatePtr cand,
                               int partType, edm::Handle<edm::ValueMap<float>> puppiWeight) {
    if (partType == 0 ){
      return (*puppiWeight)[pfCand];
    }
    if (partType == 1) {
      return packedPart.puppiWeight();
    }
    return -1;
  }

  static double getTime(const reco::PFCandidatePtr pfCand,
                        const pat::PackedCandidate packedPart,
                        const reco::CandidatePtr cand,
                        int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 1) {
      return packedPart.time();
    }
    if (partType == 0) {
      return pfCand.get()->time();
    }
    return -1;
  }

  static double getHcalEnergy_depth1(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(1);
    }
    return -1;
  }
  static double getHcalEnergy_depth2(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(2);
    }
    return -1;
  }
  static double getHcalEnergy_depth3(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(3);
    }
    return -1;
  }
  static double getHcalEnergy_depth4(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(4);
    }
    return -1;
  }
  static double getHcalEnergy_depth5(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(5);
    }
    return -1;
  }
  static double getHcalEnergy_depth6(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(6);
    }
    return -1;
  }
  static double getHcalEnergy_depth7(const reco::PFCandidatePtr pfCand,
                                     const pat::PackedCandidate packedPart,
                                     const reco::CandidatePtr cand,
                                     int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalDepthEnergyFraction(7);
    }
    return -1;
  }

  static double getEcalEnergy(const reco::PFCandidatePtr pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->ecalEnergy();
    }
    if (partType == 1) {
      return (1.0 - packedPart.hcalFraction()) * packedPart.energy();
    }
    return -1;
  }
  static double getRawEcalEnergy(const reco::PFCandidatePtr pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->rawEcalEnergy();
    }
    if (partType == 1) {
      return (1.0 - packedPart.rawHcalFraction()) * packedPart.energy();
    }
    return -1;
  }
  static double getHcalEnergy(const reco::PFCandidatePtr pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->hcalEnergy();
    }
    if (partType == 1) {
      return packedPart.hcalFraction() * packedPart.energy();
    }
    return -1;
  }
  static double getRawHcalEnergy(const reco::PFCandidatePtr pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType == 0) {
      return pfCand.get()->rawHcalEnergy();
    }
    if (partType == 1) {
      return packedPart.rawHcalFraction() * packedPart.energy();
    }
    return -1;
  }
  static double getHOEnergy(const reco::PFCandidatePtr pfCand,
                            const pat::PackedCandidate packedPart,
                            const reco::CandidatePtr cand,
                            int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->hoEnergy();
  }
  static double getRawHOEnergy(const reco::PFCandidatePtr pfCand,
                               const pat::PackedCandidate packedPart,
                               const reco::CandidatePtr cand,
                               int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->rawHoEnergy();
  }

  static double getMVAIsolated(const reco::PFCandidatePtr pfCand,
                               const pat::PackedCandidate packedPart,
                               const reco::CandidatePtr cand,
                               int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_Isolated();
  }
  static double getMVAEPi(const reco::PFCandidatePtr pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_e_pi();
  }
  static double getMVAEMu(const reco::PFCandidatePtr pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_e_mu();
  }
  static double getMVAPiMu(const reco::PFCandidatePtr pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_pi_mu();
  }
  static double getMVANothingGamma(const reco::PFCandidatePtr pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_nothing_gamma();
  }
  static double getMVANothingNH(const reco::PFCandidatePtr pfCand,
                                const pat::PackedCandidate packedPart,
                                const reco::CandidatePtr cand,
                                int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_nothing_nh();
  }
  static double getMVAGammaNH(const reco::PFCandidatePtr pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->mva_gamma_nh();
  }

  static double getDNNESigIsolated(const reco::PFCandidatePtr pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->dnn_e_sigIsolated();
  }
  static double getDNNESigNonIsolated(const reco::PFCandidatePtr pfCand,
                                      const pat::PackedCandidate packedPart,
                                      const reco::CandidatePtr cand,
                                      int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->dnn_e_sigNonIsolated();
  }
  static double getDNNEBkgNonIsolated(const reco::PFCandidatePtr pfCand,
                                      const pat::PackedCandidate packedPart,
                                      const reco::CandidatePtr cand,
                                      int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->dnn_e_bkgNonIsolated();
  }
  static double getDNNEBkgTauIsolated(const reco::PFCandidatePtr pfCand,
                                      const pat::PackedCandidate packedPart,
                                      const reco::CandidatePtr cand,
                                      int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->dnn_e_bkgTau();
  }
  static double getDNNEBkgPhotonIsolated(const reco::PFCandidatePtr pfCand,
                                         const pat::PackedCandidate packedPart,
                                         const reco::CandidatePtr cand,
                                         int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->dnn_e_bkgPhoton();
  }

  static double getECalEFrac(const reco::PFCandidatePtr pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->ecalEnergy() / pfCand.get()->energy();
  }
  static double getHCalEFrac(const reco::PFCandidatePtr pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->hcalEnergy() / pfCand.get()->energy();
  }
  static double getPS1Energy(const reco::PFCandidatePtr pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->pS1Energy();
  }
  static double getPS2Energy(const reco::PFCandidatePtr pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->pS2Energy();
  }
  static double getPSEnergy(const reco::PFCandidatePtr pfCand,
                            const pat::PackedCandidate packedPart,
                            const reco::CandidatePtr cand,
                            int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    return pfCand.get()->pS1Energy() + pfCand.get()->pS2Energy();
  }

  static double getTrackPt(const reco::PFCandidatePtr pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->pt();
    return -1;
  }

  static double getTrackNStripHits(const reco::PFCandidatePtr pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->hitPattern().numberOfValidStripHits();
    return -1;
  }

  static double getTrackNPixHits(const reco::PFCandidatePtr pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->hitPattern().numberOfValidPixelHits();

    return -1;
  }

  static double getTrackChi2(const reco::PFCandidatePtr pfCand,
                             const pat::PackedCandidate packedPart,
                             const reco::CandidatePtr cand,
                             int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->chi2();
    return -1;
  }

  static double getTrackPtError(const reco::PFCandidatePtr pfCand,
                                const pat::PackedCandidate packedPart,
                                const reco::CandidatePtr cand,
                                int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->ptError();
    return -1;
  }

  static double getTrackRelPtError(const reco::PFCandidatePtr pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->ptError() / (pfCand->trackRef())->pt();
    return -1;
  }

  static double getTrackD0(const reco::PFCandidatePtr pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->d0();
    return -1;
  }

  static double getTrackZ0(const reco::PFCandidatePtr pfCand,
                           const pat::PackedCandidate packedPart,
                           const reco::CandidatePtr cand,
                           int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->d0();
    return -1;
  }

  static double getTrackThetaError(const reco::PFCandidatePtr pfCand,
                                   const pat::PackedCandidate packedPart,
                                   const reco::CandidatePtr cand,
                                   int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->thetaError();
    return -1;
  }

  static double getTrackEtaError(const reco::PFCandidatePtr pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->etaError();
    return -1;
  }

  static double getTrackPhiError(const reco::PFCandidatePtr pfCand,
                                 const pat::PackedCandidate packedPart,
                                 const reco::CandidatePtr cand,
                                 int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    if (pfCand.get()->trackRef().isNonnull())
      return (pfCand.get()->trackRef())->phiError();
    return -1;
  }

  static double getEoverP(const reco::PFCandidatePtr pfCand,
                          const pat::PackedCandidate packedPart,
                          const reco::CandidatePtr cand,
                          int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    double energy = 0;
    int maxElement = pfCand.get()->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.get()->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.get()->elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL ||
              elements[iEle].type() == reco::PFBlockElement::ECAL) {  // Element is HB or HE
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            const reco::PFCluster& cluster = *clusterref;
            energy += cluster.energy();
          }
        }
      }
    }
    return energy / pfCand.get()->p();
  }

  static double getHCalEnergy(const reco::PFCandidatePtr pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    double energy = 0;
    int maxElement = pfCand.get()->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.get()->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.get()->elementsInBlocks()[e].second) {
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

  static double getECalEnergy(const reco::PFCandidatePtr pfCand,
                              const pat::PackedCandidate packedPart,
                              const reco::CandidatePtr cand,
                              int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    double energy = 0;
    int maxElement = pfCand.get()->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.get()->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.get()->elementsInBlocks()[e].second) {
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

  static double getNTracksInBlock(const reco::PFCandidatePtr pfCand,
                                  const pat::PackedCandidate packedPart,
                                  const reco::CandidatePtr cand,
                                  int partType, edm::Handle<edm::ValueMap<float>>) {
    if (partType) {
      return -1;
    }
    // We need this function to return a double, even though this is an integer value
    double nTrack = 0;
    int maxElement = pfCand.get()->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand.get()->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand.get()->elementsInBlocks()[e].second) {
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
  unsigned int m_runNumber;

  typedef edm::View<reco::Candidate> CandView;
  //edm::EDGetTokenT<reco::PFCandidateCollection> thePfCandidateCollection_;
  //edm::EDGetTokenT<std::vector<edm::FwdPtr<reco::PFCandidate>>> thePfCandidateCollection_;
  edm::EDGetTokenT<CandView> thePfCandidateCollection_;

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
  typedef std::vector<int> vInt;

  vstring m_triggerOptions;
  // Information on which observables to make histograms for.
  // In the config file, this should come as a comma-separated list of
  // the observable name, the number of bins for the histogram, and
  // the lowest and highest values for the histogram.
  // The observable name should have an entry in m_funcMap to define how
  // it can be retrieved from a PFCandidate.
  vstring m_pfNames;
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

  //std::vector<std::string> m_pfNames;
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

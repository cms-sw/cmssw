#ifndef PFAnalyzer_H
#define PFAnalyzer_H

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

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

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
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "fastjet/PseudoJet.hh"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
class PFAnalyzer;

class PFAnalyzer : public DQMEDAnalyzer {
public:
  /// Constructor
  PFAnalyzer(const edm::ParameterSet&);

  /// Destructor
  ~PFAnalyzer() override;

  /// Initialize parameters for histo binning
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// Initialize run-based parameters
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct binInfo;
  
  // The input collections are read polymorphically, so that the same code can
  // handle RECO / AOD / HLT (reco::PFCandidate / reco::PFJet) and miniAOD
  // (pat::PackedCandidate / pat::Jet) inputs without duplicating anything.
  typedef edm::View<reco::Candidate> CandView;
  typedef edm::View<reco::Jet> JetView;

  // Resolve the concrete type hiding behind a generic candidate pointer.
  // Observables that are only defined for one of the two representations use
  // these to check whether they are applicable.
  static const reco::PFCandidate* asPF(const reco::CandidatePtr& cand) {
    return dynamic_cast<const reco::PFCandidate*>(cand.get());
  }
  static const pat::PackedCandidate* asPacked(const reco::CandidatePtr& cand) {
    return dynamic_cast<const pat::PackedCandidate*>(cand.get());
  }

  // The equivalent of reco::PFCandidate::particleId(), but usable for any
  // reco::Candidate. reco::PFCandidate::particleId() is itself defined as
  // translatePdgIdToType(pdgId()), which only ever looks at the pdgId, so this
  // gives the same answer for PF candidates and extends it to packed ones.
  static reco::PFCandidate::ParticleType particleType(const reco::CandidatePtr& cand) {
    if (!cand)
      return reco::PFCandidate::ParticleType::X;
    switch (std::abs(cand->pdgId())) {
      case 211:
        return reco::PFCandidate::ParticleType::h;
      case 11:
        return reco::PFCandidate::ParticleType::e;
      case 13:
        return reco::PFCandidate::ParticleType::mu;
      case 22:
        return reco::PFCandidate::ParticleType::gamma;
      case 130:
        return reco::PFCandidate::ParticleType::h0;
      case 1:
        return reco::PFCandidate::ParticleType::h_HF;
      case 2:
        return reco::PFCandidate::ParticleType::egamma_HF;
      default:
        return reco::PFCandidate::ParticleType::X;
    }
  }

  // The puppi weight comes either from an external ValueMap (RECO/HLT) or from
  // the candidate itself (miniAOD). Returns -1 when it is not available, which
  // includes the cases where the ValueMap was not produced and where it was
  // keyed on a different collection than the one being read here.
  static double puppiWeightOf(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>& puppiWeight) {
    if (puppiWeight.isValid() && puppiWeight->contains(cand.id()))
      return (*puppiWeight)[cand];
    if (const pat::PackedCandidate* packedPart = asPacked(cand))
      return packedPart->puppiWeight();
    return -1;
  }

  // A map between an observable name and a function that obtains that observable from a  PFCandidate.
  // This allows us to construct more complicated observables easily, and have it more configurable
  // in the config file.
  std::map<std::string, std::function<double(const reco::CandidatePtr&, const edm::Handle<edm::ValueMap<float>>&)>>
      m_funcMap;
  std::map<std::string,
           std::function<double(const std::vector<reco::CandidatePtr>& pfCands, reco::PFCandidate::ParticleType pfType)>>
      m_eventFuncMap;

  std::map<std::string,
           std::function<double(
               const std::vector<reco::CandidatePtr>& pfCands, reco::PFCandidate::ParticleType pfType, const reco::Jet&)>>
      m_jetWideFuncMap;

  std::map<std::string, std::function<double(const reco::CandidatePtr&, const reco::Jet&)>> m_pfInJetFuncMap;
  std::map<std::string, std::function<double(const reco::Jet&, const std::vector<reco::CandidatePtr>& pfCands)>>
      m_jetFuncMap;

  std::map<std::string, std::function<bool(const JetView& pfJets)>> m_eventSelectionMap;

  binInfo getBinInfo(std::string);

  int getPFBin(const reco::CandidatePtr& cand, unsigned int i, const edm::Handle<edm::ValueMap<float>>& puppiWeight);
  int getJetBin(const reco::Jet& jetCand, const std::vector<reco::CandidatePtr>& pfCands, unsigned int i);

  int getBinNumber(double binVal, std::vector<double> bins);
  int getBinNumbers(std::vector<double> binVal, std::vector<std::vector<double>> bins);
  std::vector<double> getBinList(std::string binString);

  std::vector<std::string> getAllSuffixes(std::vector<std::string> observables,
                                          std::vector<std::vector<double>> binnings);
  std::string stringWithDecimals(int bin, std::vector<double> bins);

  std::string getSuffix(std::vector<int> binList,
                        std::vector<std::string> observables,
                        std::vector<std::vector<double>> binnings);

  static double getEnergySpectrum(const reco::CandidatePtr& cand, const reco::Jet& jet) {
    if (!jet.pt())
      return -1;
    return cand->pt() / jet.pt();
  }

  static double getNPFC(const std::vector<reco::CandidatePtr>& pfCands, reco::PFCandidate::ParticleType pfType) {
    int nPF = 0;
    for (const auto& pfCand : pfCands) {
      // We use X to indicate all
      if (particleType(pfCand) == pfType || pfType == reco::PFCandidate::ParticleType::X) {
        nPF++;
      }
    }
    return nPF;
  }

  static double getNPFCinJet(const std::vector<reco::CandidatePtr>& pfCands,
                             reco::PFCandidate::ParticleType pfType,
                             const reco::Jet& jet) {
    int nPF = 0;
    for (const auto& pfCand : pfCands) {
      if (!pfCand)
        continue;
      // We use X to indicate all
      if (particleType(pfCand) == pfType || pfType == reco::PFCandidate::ParticleType::X)
        nPF++;
    }
    return nPF;
  }

  static double getMaxPt(const std::vector<reco::CandidatePtr>& pfCands, reco::PFCandidate::ParticleType pfType) {
    double maxPt = 0;
    for (const auto& pfCand : pfCands) {
      if (!pfCand)
        continue;
      // We use X to indicate all
      if (particleType(pfCand) == pfType || pfType == reco::PFCandidate::ParticleType::X) {
        if (pfCand->pt() > maxPt)
          maxPt = pfCand->pt();
      }
    }
    return maxPt;
  }

  static double getMaxPtFracJet(const std::vector<reco::CandidatePtr>& pfCands,
                                reco::PFCandidate::ParticleType pfType,
                                const reco::Jet& jet) {
    if (!jet.pt())
      return -1;
    double maxPt = 0;
    for (const auto& pfCand : pfCands) {
      if (!pfCand)
        continue;
      // We use X to indicate all
      if (particleType(pfCand) == pfType || pfType == reco::PFCandidate::ParticleType::X)
        if (pfCand->pt() > maxPt)
          maxPt = pfCand->pt();
    }
    return maxPt / jet.pt();
  }

  // Various functions designed to get information from a PF canddidate.
  // The kinematic ones only need the reco::Candidate interface, and so are
  // valid whatever the concrete type of the candidate is.
  static double getPt(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) { return cand->pt(); }

  static double getLogPt(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return cand->pt() > 0 ? log10(cand->pt()) : -10;
  }

  static double getEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return cand->energy();
  }
  static double getEta(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) { return cand->eta(); }
  static double getAbsEta(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return std::abs(cand->eta());
  }
  static double getPhi(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) { return cand->phi(); }

  static double getHadCalibration(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand)
      return -1;
    if (pfCand->rawHcalEnergy() == 0) {
      return -1;
    }
    return pfCand->hcalEnergy() / pfCand->rawHcalEnergy();
  }
  static double getPuppiWeight(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>& puppiWeight) {
    return puppiWeightOf(cand, puppiWeight);
  }
  static double getPuppiPt(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>& puppiWeight) {
    double weight = puppiWeightOf(cand, puppiWeight);
    if (weight < 0)
      return -1;
    return weight * cand->pt();
  }

  static double getLogPuppiPt(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>& puppiWeight) {
    double weight = puppiWeightOf(cand, puppiWeight);
    if (weight < 0)
      return -1;
    return weight * cand->pt() > 0 ? log10(weight * cand->pt()) : -10;
  }

  static double getPuppiEta(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>& puppiWeight) {
    double weight = puppiWeightOf(cand, puppiWeight);
    if (weight < 0)
      return -1;
    fastjet::PseudoJet weightedPF =
        fastjet::PseudoJet(weight * cand->px(), weight * cand->py(), weight * cand->pz(), weight * cand->energy());
    return weightedPF.eta();
  }

  static double getTime(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return packedPart->time();
    }
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->time();
    }
    return -1;
  }

  static double getHcalEnergyDepth(const reco::CandidatePtr& cand, unsigned int depth) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand)
      return -1;
    return pfCand->hcalDepthEnergyFraction(depth);
  }

  static double getHcalEnergy_depth1(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 1);
  }
  static double getHcalEnergy_depth2(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 2);
  }
  static double getHcalEnergy_depth3(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 3);
  }
  static double getHcalEnergy_depth4(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 4);
  }
  static double getHcalEnergy_depth5(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 5);
  }
  static double getHcalEnergy_depth6(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 6);
  }
  static double getHcalEnergy_depth7(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    return getHcalEnergyDepth(cand, 7);
  }

  static double getEcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->ecalEnergy();
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return (1.0 - packedPart->hcalFraction()) * packedPart->energy();
    }
    return -1;
  }
  static double getRawEcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->rawEcalEnergy();
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return (1.0 - packedPart->rawHcalFraction()) * packedPart->energy();
    }
    return -1;
  }
  static double getHcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->hcalEnergy();
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return packedPart->hcalFraction() * packedPart->energy();
    }
    return -1;
  }
  static double getRawHcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->rawHcalEnergy();
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return packedPart->rawHcalFraction() * packedPart->energy();
    }
    return -1;
  }
  static double getHOEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->hoEnergy();
  }
  static double getRawHOEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->rawHoEnergy();
  }

  static double getRelEcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->ecalEnergy() / pfCand->energy();
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return (1.0 - packedPart->hcalFraction()) * packedPart->energy();
    }
    return -1;
  }
  static double getRelRawEcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->rawEcalEnergy() / (pfCand->rawHoEnergy() + pfCand->rawHcalEnergy() + pfCand->rawEcalEnergy());
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return (1.0 - packedPart->rawHcalFraction()) * packedPart->energy();
    }
    return -1;
  }
  static double getRelHcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->hcalEnergy() / pfCand->energy();
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return packedPart->hcalFraction() * packedPart->energy();
    }
    return -1;
  }
  static double getRelRawHcalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    if (const reco::PFCandidate* pfCand = asPF(cand)) {
      return pfCand->rawHcalEnergy() / (pfCand->rawHoEnergy() + pfCand->rawHcalEnergy() + pfCand->rawEcalEnergy());
    }
    if (const pat::PackedCandidate* packedPart = asPacked(cand)) {
      return packedPart->rawHcalFraction() * packedPart->energy();
    }
    return -1;
  }

  static double getRelHOEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->hoEnergy() / pfCand->energy();
  }
  static double getRelRawHOEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->rawHoEnergy() / (pfCand->rawHoEnergy() + pfCand->rawHcalEnergy() + pfCand->rawEcalEnergy());
  }

  static double getMVAIsolated(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_Isolated();
  }
  static double getMVAEPi(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_e_pi();
  }
  static double getMVAEMu(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_e_mu();
  }
  static double getMVAPiMu(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_pi_mu();
  }
  static double getMVANothingGamma(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_nothing_gamma();
  }
  static double getMVANothingNH(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_nothing_nh();
  }
  static double getMVAGammaNH(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->mva_gamma_nh();
  }

  static double getDNNESigIsolated(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->dnn_e_sigIsolated();
  }
  static double getDNNESigNonIsolated(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->dnn_e_sigNonIsolated();
  }
  static double getDNNEBkgNonIsolated(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->dnn_e_bkgNonIsolated();
  }
  static double getDNNEBkgTauIsolated(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->dnn_e_bkgTau();
  }
  static double getDNNEBkgPhotonIsolated(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->dnn_e_bkgPhoton();
  }

  static double getECalEFrac(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->ecalEnergy() / pfCand->energy();
  }
  static double getHCalEFrac(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->hcalEnergy() / pfCand->energy();
  }
  static double getPS1Energy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->pS1Energy();
  }
  static double getPS2Energy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->pS2Energy();
  }
  static double getPSEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    return pfCand->pS1Energy() + pfCand->pS2Energy();
  }

  static double getTrackPt(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->pt();
    return -1;
  }

  static double getTrackNStripHits(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->hitPattern().numberOfValidStripHits();
    return -1;
  }

  static double getTrackNPixHits(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->hitPattern().numberOfValidPixelHits();

    return -1;
  }

  static double getTrackChi2(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->chi2();
    return -1;
  }

  static double getTrackPtError(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->ptError();
    return -1;
  }

  static double getTrackRelPtError(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->ptError() / (pfCand->trackRef())->pt();
    return -1;
  }

  static double getTrackD0(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->d0();
    return -1;
  }

  static double getTrackDZ(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->dz();
    return -1;
  }

  static double getTrackThetaError(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->thetaError();
    return -1;
  }

  static double getTrackEtaError(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->etaError();
    return -1;
  }

  static double getTrackPhiError(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    if (pfCand->trackRef().isNonnull())
      return (pfCand->trackRef())->phiError();
    return -1;
  }

  static double getEoverP(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    double energy = 0;
    int maxElement = pfCand->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand->elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL ||
              elements[iEle].type() == reco::PFBlockElement::ECAL) {  // Element is HB or HE
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            const reco::PFCluster& cluster = *clusterref;
            energy += cluster.energy();
          }
        }
      }
    }
    return energy / pfCand->p();
  }

  static double getHCalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    double energy = 0;
    int maxElement = pfCand->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand->elementsInBlocks()[e].second) {
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

  static double getECalEnergy(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    double energy = 0;
    int maxElement = pfCand->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand->elementsInBlocks()[e].second) {
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

  static double getNTracksInBlock(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    // We need this function to return a double, even though this is an integer value
    double nTrack = 0;
    int maxElement = pfCand->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand->elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::TRACK) {  // Element is HB or HE
            nTrack += 1;
          }
        }
      }
    }
    return nTrack;
  }

  static double getCellsInBlock(const reco::CandidatePtr& cand, const edm::Handle<edm::ValueMap<float>>&) {
    const reco::PFCandidate* pfCand = asPF(cand);
    if (!pfCand) {
      return -1;
    }
    // We need this function to return a double, even though this is an integer value
    double nTrack = 0;
    int maxElement = pfCand->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = pfCand->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == pfCand->elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL) {  // Element is HB or HE
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            const reco::PFCluster& cluster = *clusterref;

            nTrack += cluster.recHitFractions().size();
          }
        }
      }
    }
    return nTrack;
  }

  static double getJetPt(const reco::Jet& jet, const std::vector<reco::CandidatePtr>& pfCands) { return jet.pt(); }
  static double getJetChargeFrac(const reco::Jet& jet, const std::vector<reco::CandidatePtr>& pfCands) {
    if (!jet.pt())
      return -1;
    double chargeFrac = 0;

    for (const auto& recoPF : pfCands) {
      if (!recoPF)
        continue;
      reco::PFCandidate::ParticleType pfType = particleType(recoPF);
      if (pfType == reco::PFCandidate::ParticleType::h || pfType == reco::PFCandidate::ParticleType::e ||
          pfType == reco::PFCandidate::ParticleType::mu)
        chargeFrac += recoPF->pt();
    }
    return chargeFrac / jet.pt();
  }

  bool passesTriggerSelection(const JetView& pfJets,
                              const edm::Handle<edm::TriggerResults>& triggerResults,
                              const edm::TriggerNames& triggerNames,
                              const std::vector<std::string> triggerOptions) {
    // Hack to make it pass the lowest unprescaled HLT?
    Int_t JetHiPass = 0;

    for (unsigned i = 0; i < triggerNames.size(); ++i) {
      for (unsigned j = 0; j < triggerOptions.size(); ++j) {
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

  static bool passesNoCutSelection(const JetView& pfJets) { return true; }

  static bool passesDijetSelection(const JetView& pfJets) {
    if (pfJets.size() < 2)
      return false;
    if (pfJets[0].pt() < 450)
      return false;
    if (!pfJets[1].pt())
      return false;
    if (pfJets[0].pt() / pfJets[1].pt() > 2)
      return false;

    return true;
  }

  static bool passesAnomalousSelection(const JetView& pfJets) {
    if (pfJets.size() < 2)
      return false;
    if (pfJets[0].pt() < 450)
      return false;
    if (pfJets[1].pt() / pfJets[0].pt() > 0.5)
      return false;

    return true;
  }

  bool m_isHLT;
  bool m_isMiniAOD;
  unsigned int m_runNumber;

  edm::EDGetTokenT<CandView> pfCandidateToken_;
  edm::EDGetTokenT<JetView> jetsToken_;

  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;
  edm::InputTag srcWeights;

  edm::EDGetTokenT<edm::ValueMap<float>> puppiWeightToken_;

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

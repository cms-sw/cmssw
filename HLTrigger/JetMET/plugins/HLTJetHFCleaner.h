#ifndef HLTrigger_JetMET_plugins_HLTJetHFCleaner_h
#define HLTrigger_JetMET_plugins_HLTJetHFCleaner_h

// -*- C++ -*-
//
// Package:    HLTrigger/JetMET
// Class:      HLTJetHFCleaner
//
/**\class HLTJetHFCleaner HLTJetHFCleaner.cc HLTrigger/HLTJetHFCleaner/plugins/HLTJetHFCleaner.cc

 Description: Cleaning module that produces a cleaned set of jets after applying forward jet cuts.

 Implementation:
    This module creates a new HF-cleaned jet collection by skimming out the forward jets that fail the forward jet cuts. 
    Only the forward high pt jets (|eta| > etaMin_ & pt > jetPtMin_) are considered for the cuts.
    The cuts are based on the following HF shape variables: sigmaEtaEta, sigmaPhiPhi and centralStripSize.
    
    This module applies the following cuts to such jets:
      sigmaEtaEta - sigmaPhiPhi <= sigmaEtaPhiDiffMax_
      centralStripSize <= centralEtaStripSizeMax_
    
    In addition, forward jets in the corner-region phase space of sigmaEtaEta-sigmaPhiPhi phase space, 
    such that:
      (sigmaEtaEta < cornerCutSigmaEtaEta_) & (sigmaPhiPhi < cornerCutSigmaPhiPhi_)
    the jets will be discarded as well.

    All cuts (and whether to apply the cuts) can be configured using the PSet interface of this module.

*/
//
// Original Author:  Alp Akpinar
//         Created:  Wed, 01 Dec 2021 15:47:19 GMT
//
//

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/METReco/interface/MET.h"

template <typename JetType>
class HLTJetHFCleaner : public edm::global::EDProducer<> {
public:
  typedef std::vector<JetType> JetCollection;
  typedef edm::Ref<JetCollection> JetRef;
  explicit HLTJetHFCleaner(const edm::ParameterSet&);
  ~HLTJetHFCleaner() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<std::vector<JetType>> jetToken_;
  const edm::EDGetTokenT<edm::View<reco::MET>> metToken_;

  const edm::EDGetTokenT<edm::ValueMap<float>> sigmaEtaEtaToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> sigmaPhiPhiToken_;
  const edm::EDGetTokenT<edm::ValueMap<int>> centralEtaStripSizeToken_;

  // Minimum jet pt to consider while applying HF cuts
  const float jetPtMin_;
  const float dphiJetMetMin_;

  // Cuts will be applied to jets only with jetEtaMin_ < |eta| < jetEtaMax_
  const float jetEtaMin_;
  const float jetEtaMax_;

  const float sigmaEtaPhiDiffMax_;
  const float centralEtaStripSizeMax_;
  const float cornerCutSigmaEtaEta_;
  const float cornerCutSigmaPhiPhi_;
  const bool applySigmaEtaPhiCornerCut_;
  const bool applySigmaEtaPhiCut_;
  const bool applyStripSizeCut_;
};

template <typename JetType>
HLTJetHFCleaner<JetType>::HLTJetHFCleaner(const edm::ParameterSet& iConfig)
    : jetToken_(consumes(iConfig.getParameter<edm::InputTag>("jets"))),
      metToken_(consumes(iConfig.getParameter<edm::InputTag>("mets"))),
      sigmaEtaEtaToken_(consumes(iConfig.getParameter<edm::InputTag>("sigmaEtaEta"))),
      sigmaPhiPhiToken_(consumes(iConfig.getParameter<edm::InputTag>("sigmaPhiPhi"))),
      centralEtaStripSizeToken_(consumes(iConfig.getParameter<edm::InputTag>("centralEtaStripSize"))),
      jetPtMin_(iConfig.getParameter<double>("jetPtMin")),
      dphiJetMetMin_(iConfig.getParameter<double>("dphiJetMetMin")),
      jetEtaMin_(iConfig.getParameter<double>("jetEtaMin")),
      jetEtaMax_(iConfig.getParameter<double>("jetEtaMax")),
      sigmaEtaPhiDiffMax_(iConfig.getParameter<double>("sigmaEtaPhiDiffMax")),
      centralEtaStripSizeMax_(iConfig.getParameter<int>("centralEtaStripSizeMax")),
      cornerCutSigmaEtaEta_(iConfig.getParameter<double>("cornerCutSigmaEtaEta")),
      cornerCutSigmaPhiPhi_(iConfig.getParameter<double>("cornerCutSigmaPhiPhi")),
      applySigmaEtaPhiCornerCut_(iConfig.getParameter<bool>("applySigmaEtaPhiCornerCut")),
      applySigmaEtaPhiCut_(iConfig.getParameter<bool>("applySigmaEtaPhiCut")),
      applyStripSizeCut_(iConfig.getParameter<bool>("applyStripSizeCut")) {
  produces<JetCollection>();
}

template <typename JetType>
void HLTJetHFCleaner<JetType>::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto const jets = iEvent.getHandle(jetToken_);
  auto const& mets = iEvent.get(metToken_);

  // The set of cleaned jets
  auto cleaned_jets = std::make_unique<JetCollection>();
  cleaned_jets->reserve(jets->size());

  // Check if length of the MET vector is 1, throw exception otherwise
  if (mets.size() != 1) {
    throw cms::Exception("ComparisonFailure") << "Size of reco::MET collection is different from 1";
  }
  auto const met_phi = mets.front().phi();

  // HF shape variables gathered from valueMaps
  // Normally, these valueMaps are the output of the HFJetShowerShape module:
  // https://github.com/cms-sw/cmssw/blob/master/RecoJets/JetProducers/plugins/HFJetShowerShape.cc
  auto const& sigmaEtaEtas = iEvent.get(sigmaEtaEtaToken_);
  auto const& sigmaPhiPhis = iEvent.get(sigmaPhiPhiToken_);
  auto const& centralEtaStripSizes = iEvent.get(centralEtaStripSizeToken_);

  // Loop over the jets and do cleaning
  for (uint ijet = 0; ijet < jets->size(); ijet++) {
    auto const jet = (*jets)[ijet];
    auto const withinEtaRange = (std::abs(jet.eta()) < jetEtaMax_) and (std::abs(jet.eta()) > jetEtaMin_);
    // Cuts are applied to jets that are back to back with MET
    if (std::abs(reco::deltaPhi(jet.phi(), met_phi)) <= dphiJetMetMin_) {
      cleaned_jets->emplace_back(jet);
      continue;
    }
    // If the jet is outside the range that we're interested in,
    // append it to the cleaned jets collection and continue
    if ((jet.pt() < jetPtMin_) or (!withinEtaRange)) {
      cleaned_jets->emplace_back(jet);
      continue;
    }

    JetRef const jetRef(jets, ijet);

    // Cuts related to eta and phi width of HF showers:
    // Sigma eta-phi based cut + central strip size cut
    // We'll apply the sigma eta+phi dependent cut only if requested via the config file
    if (applySigmaEtaPhiCut_) {
      // Sigma eta eta and sigma phi phi
      auto const sigmaEtaEta = sigmaEtaEtas[jetRef];
      auto const sigmaPhiPhi = sigmaPhiPhis[jetRef];

      // Check if sigma eta eta and sigma phi phi are both set to -1.
      // If this is the case, we can skip the jet without evaluating any mask.
      if ((sigmaEtaEta < 0) and (sigmaPhiPhi < 0)) {
        cleaned_jets->emplace_back(jet);
        continue;
      }

      auto passSigmaEtaPhiCut = (sigmaEtaEta - sigmaPhiPhi) <= sigmaEtaPhiDiffMax_;

      if (applySigmaEtaPhiCornerCut_) {
        auto const inCorner = (sigmaEtaEta < cornerCutSigmaEtaEta_) and (sigmaPhiPhi < cornerCutSigmaPhiPhi_);
        passSigmaEtaPhiCut &= !inCorner;
      }
      if (!passSigmaEtaPhiCut) {
        continue;
      }
    }

    // Cut related to central strip size
    if (applyStripSizeCut_ and (centralEtaStripSizes[jetRef] > centralEtaStripSizeMax_)) {
      continue;
    }

    cleaned_jets->emplace_back(jet);
  }

  iEvent.put(std::move(cleaned_jets));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename JetType>
void HLTJetHFCleaner<JetType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("hltAK4PFJetsTightIDCorrected"))->setComment("Input jet collection.");
  desc.add<edm::InputTag>("mets", edm::InputTag("hltMet"))->setComment("Input MET collection.");
  desc.add<edm::InputTag>("sigmaEtaEta", edm::InputTag("hltHFJetShowerShape", "sigmaEtaEta"))
      ->setComment("Input collection which stores the sigmaEtaEta values per jet.");
  desc.add<edm::InputTag>("sigmaPhiPhi", edm::InputTag("hltHFJetShowerShape", "sigmaPhiPhi"))
      ->setComment("Input collection which stores the sigmaPhiPhis values per jet.");
  desc.add<edm::InputTag>("centralEtaStripSize", edm::InputTag("hltHFJetShowerShape", "centralEtaStripSize"))
      ->setComment("Input collection which stores the central strip size values per jet.");
  desc.add<double>("jetPtMin", 100)->setComment("The minimum pt value for a jet such that the cuts will be applied.");
  desc.add<double>("dphiJetMetMin", 2.5)
      ->setComment(
          "The minimum value for deltaPhi between jet and MET, such that the cuts will be applied to a given jet.");
  desc.add<double>("jetEtaMin", 2.9)->setComment("Minimum value of jet |eta| for which the cuts will be applied.");
  desc.add<double>("jetEtaMax", 5.0)->setComment("Maximum value of jet |eta| for which the cuts will be applied.");
  desc.add<double>("sigmaEtaPhiDiffMax", 0.05)
      ->setComment("Determines the threshold in the following cut: sigmaEtaEta-sigmaPhiPhi <= X");
  desc.add<double>("cornerCutSigmaEtaEta", 0.02)
      ->setComment(
          "Corner cut value for sigmaEtaEta. Jets will be cut if both sigmaEtaEta and sigmaPhiPhi are lower than the "
          "corner value.");
  desc.add<double>("cornerCutSigmaPhiPhi", 0.02)
      ->setComment(
          "Corner cut value for sigmaPhiPhi. Jets will be cut if both sigmaEtaEta and sigmaPhiPhi are lower than the "
          "corner value.");
  desc.add<int>("centralEtaStripSizeMax", 2)
      ->setComment("Determines the threshold in the following cut: centralEtaStripSize <= X");
  desc.add<bool>("applySigmaEtaPhiCornerCut", true)->setComment("Boolean specifying whether to apply the corner cut.");
  desc.add<bool>("applySigmaEtaPhiCut", true)
      ->setComment("Boolean specifying whether to apply the sigmaEtaEta-sigmaPhiPhi cut.");
  desc.add<bool>("applyStripSizeCut", true)
      ->setComment("Boolean specifying whether to apply the centralEtaStripSize cut.");
  descriptions.addWithDefaultLabel(desc);
}

#endif

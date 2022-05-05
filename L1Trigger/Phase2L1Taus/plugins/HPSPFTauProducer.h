#ifndef L1Trigger_Phase2L1Taus_HPSPFTauProducer_h
#define L1Trigger_Phase2L1Taus_HPSPFTauProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "L1Trigger/Phase2L1Taus/interface/L1HPSPFTauQualityCut.h"  // L1HPSPFTauQualityCut
#include "L1Trigger/Phase2L1Taus/interface/L1HPSPFTauBuilder.h"     // L1HPSPFTauBuilder
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"         // l1t::HPSPFTau
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"      // l1t::HPSPFTauCollection
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"  // l1t::PFCandidate, l1t::PFCandidateCollection, l1t::PFCandidateRef
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

#include <string>
#include <vector>

class HPSPFTauProducer : public edm::stream::EDProducer<> {
public:
  explicit HPSPFTauProducer(const edm::ParameterSet& cfg);
  ~HPSPFTauProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  std::string moduleLabel_;

  L1HPSPFTauBuilder tauBuilder_;

  edm::InputTag srcL1PFCands_;
  edm::EDGetTokenT<l1t::PFCandidateCollection> tokenL1PFCands_;
  edm::InputTag srcL1Jets_;
  edm::EDGetTokenT<std::vector<reco::CaloJet>> tokenL1Jets_;
  edm::InputTag srcL1Vertices_;
  edm::EDGetTokenT<std::vector<l1t::TkPrimaryVertex>> tokenL1Vertices_;

  std::vector<L1HPSPFTauQualityCut> signalQualityCutsDzCutDisabled_;
  std::vector<L1HPSPFTauQualityCut> isolationQualityCutsDzCutDisabled_;

  bool useChargedPFCandSeeds_;
  double minSeedChargedPFCandPt_;
  double maxSeedChargedPFCandEta_;
  double maxSeedChargedPFCandDz_;

  bool useJetSeeds_;
  double minSeedJetPt_;
  double maxSeedJetEta_;

  double minPFTauPt_;
  double maxPFTauEta_;
  double minLeadChargedPFCandPt_;
  double maxLeadChargedPFCandEta_;
  double maxLeadChargedPFCandDz_;
  double maxChargedIso_;
  double maxChargedRelIso_;

  double deltaRCleaning_;
  double deltaR2Cleaning_;

  bool applyPreselection_;

  bool debug_;
  const double isPFTauPt_ = 1.;
};

#endif

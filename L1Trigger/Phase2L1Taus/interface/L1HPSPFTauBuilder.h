#ifndef L1Trigger_Phase2L1Taus_L1HPSPFTauBuilder_h
#define L1Trigger_Phase2L1Taus_L1HPSPFTauBuilder_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"             // edm::ParameterSet
#include "DataFormats/Provenance/interface/ProductID.h"             // edm::ProductID
#include "L1Trigger/Phase2L1Taus/interface/L1HPSPFTauQualityCut.h"  // L1HPSPFTauQualityCut
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"         // l1t::HPSPFTau
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"  // l1t::PFCandidate, l1t::PFCandidateCollection, l1t::PFCandidateRef
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include <vector>

class L1HPSPFTauBuilder {
public:
  L1HPSPFTauBuilder(const edm::ParameterSet& cfg);
  ~L1HPSPFTauBuilder() = default;

  void reset();
  void setL1PFCandProductID(const edm::ProductID& l1PFCandProductID);
  void setVertex(const l1t::VertexWordRef& primaryVertex);
  void setL1PFTauSeed(const l1t::PFCandidateRef& l1PFCandSeed);
  //void setL1PFTauSeed(const reco::CaloJetRef& l1Jet_seed);
  void setL1PFTauSeed(const reco::CaloJetRef& l1JetSeed, const std::vector<l1t::PFCandidateRef>& l1PFCands);
  void addL1PFCandidates(const std::vector<l1t::PFCandidateRef>& l1PFCands);
  void buildL1PFTau();

  l1t::HPSPFTau getL1PFTau() const { return l1PFTau_; }

private:
  l1t::PFCandidateRefVector convertToRefVector(const std::vector<l1t::PFCandidateRef>& l1PFCands);

  bool isWithinSignalCone(const l1t::PFCandidate& l1PFCand);
  bool isWithinStrip(const l1t::PFCandidate& l1PFCand);
  bool isWithinIsolationCone(const l1t::PFCandidate& l1PFCand);

  reco::FormulaEvaluator signalConeSizeFormula_;

  double signalConeSize_;
  double signalConeSize2_;
  double minSignalConeSize_;
  double maxSignalConeSize_;

  bool useStrips_;
  double stripSizeEta_;
  double stripSizePhi_;

  double isolationConeSize_;
  double isolationConeSize2_;

  std::vector<L1HPSPFTauQualityCut> signalQualityCutsDzCutDisabled_;
  std::vector<L1HPSPFTauQualityCut> signalQualityCutsDzCutEnabledPrimary_;
  std::vector<L1HPSPFTauQualityCut> isolationQualityCutsDzCutDisabled_;
  std::vector<L1HPSPFTauQualityCut> isolationQualityCutsDzCutEnabledPrimary_;
  std::vector<L1HPSPFTauQualityCut> isolationQualityCutsDzCutEnabledPileup_;
  edm::ProductID l1PFCandProductID_;
  bool isPFCandSeeded_;
  l1t::PFCandidateRef l1PFCandSeed_;
  bool isJetSeeded_;
  reco::CaloJetRef l1JetSeed_;
  double l1PFTauSeedEta_;
  double l1PFTauSeedPhi_;
  double l1PFTauSeedZVtx_;
  double sumAllL1PFCandidatesPt_;
  l1t::VertexWordRef primaryVertex_;
  l1t::HPSPFTau l1PFTau_;

  reco::Particle::LorentzVector stripP4_;

  std::vector<l1t::PFCandidateRef> signalAllL1PFCandidates_;
  std::vector<l1t::PFCandidateRef> signalChargedHadrons_;
  std::vector<l1t::PFCandidateRef> signalElectrons_;
  std::vector<l1t::PFCandidateRef> signalNeutralHadrons_;
  std::vector<l1t::PFCandidateRef> signalPhotons_;
  std::vector<l1t::PFCandidateRef> signalMuons_;

  std::vector<l1t::PFCandidateRef> stripAllL1PFCandidates_;
  std::vector<l1t::PFCandidateRef> stripElectrons_;
  std::vector<l1t::PFCandidateRef> stripPhotons_;

  std::vector<l1t::PFCandidateRef> isoAllL1PFCandidates_;
  std::vector<l1t::PFCandidateRef> isoChargedHadrons_;
  std::vector<l1t::PFCandidateRef> isoElectrons_;
  std::vector<l1t::PFCandidateRef> isoNeutralHadrons_;
  std::vector<l1t::PFCandidateRef> isoPhotons_;
  std::vector<l1t::PFCandidateRef> isoMuons_;

  std::vector<l1t::PFCandidateRef> sumAllL1PFCandidates_;
  std::vector<l1t::PFCandidateRef> sumChargedHadrons_;
  std::vector<l1t::PFCandidateRef> sumElectrons_;
  std::vector<l1t::PFCandidateRef> sumNeutralHadrons_;
  std::vector<l1t::PFCandidateRef> sumPhotons_;
  std::vector<l1t::PFCandidateRef> sumMuons_;

  double sumChargedIsoPileup_;

  bool debug_;
};

#endif

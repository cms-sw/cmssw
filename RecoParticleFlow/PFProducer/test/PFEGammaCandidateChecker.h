#ifndef RecoParticleFlow_PFEGammaCandidateChecker_
#define RecoParticleFlow_PFEGammaCandidateChecker_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

/**\class PFEGammaCandidateChecker 
\brief Checks what a re-reco changes in PFCandidates.

\author Patrick Janot
\date   August 2011
*/

class PFEGammaCandidateChecker : public edm::one::EDAnalyzer<> {
public:
  explicit PFEGammaCandidateChecker(const edm::ParameterSet&);

  ~PFEGammaCandidateChecker() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void printJets(const reco::PFJetCollection& pfJetsReco, const reco::PFJetCollection& pfJetsReReco) const;

  void printMet(const reco::PFCandidateCollection& pfReco, const reco::PFCandidateCollection& pfReReco) const;

  void printElementsInBlocks(const reco::PFCandidate& cand, std::ostream& out = std::cout) const;

  /// PFCandidates in which we'll look for pile up particles
  const edm::EDGetTokenT<reco::PFCandidateCollection> tokenPFCandidatesReco_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> tokenPFCandidatesReReco_;
  const edm::EDGetTokenT<reco::PFJetCollection> tokenPFJetsReco_;
  const edm::EDGetTokenT<reco::PFJetCollection> tokenPFJetsReReco_;

  /// Cuts for comparison
  const double deltaEMax_;
  const double deltaEtaMax_;
  const double deltaPhiMax_;

  /// verbose ?
  const bool verbose_;

  /// print the blocks associated to a given candidate ?
  const bool printBlocks_;

  /// rank the candidates by Pt
  const bool rankByPt_;

  /// Counter
  unsigned entry_;

  static bool greaterPt(const reco::PFCandidate& a, const reco::PFCandidate& b) { return (a.pt() > b.pt()); }
};

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(PFEGammaCandidateChecker);

#endif

#ifndef RecoParticleFlow_PFPatProducer_PFCandidateAnalyzer_
#define RecoParticleFlow_PFPatProducer_PFCandidateAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

/**\class PFCandidateAnalyzer 
\brief produces IsolatedPFCandidates from PFCandidates

\author Colin Bernet
\date   february 2008
*/




class PFCandidateAnalyzer : public edm::EDAnalyzer {
 public:

  explicit PFCandidateAnalyzer(const edm::ParameterSet&);

  ~PFCandidateAnalyzer();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);

 private:
  

  void printElementsInBlocks(const reco::PFCandidate& cand,
			     std::ostream& out=std::cout) const;


  
  /// PFCandidates in which we'll look for pile up particles 
  edm::InputTag   inputTagPFCandidates_;
  
  /// verbose ?
  bool   verbose_;

  /// print the blocks associated to a given candidate ?
  bool   printBlocks_;

  /// rank the candidates by Pt
  bool rankByPt_;

  static bool greaterPt( const reco::PFCandidate& a, const reco::PFCandidate& b ) {
    return (a.pt()>b.pt());
  }



};

#endif

#ifndef RecoParticleFlow_PFPatProducer_PFCandidateAnalyzer_
#define RecoParticleFlow_PFPatProducer_PFCandidateAnalyzer_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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

  virtual void beginJob(const edm::EventSetup & c);

 private:
  
  void 
    fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
			     const edm::InputTag& tag, 
			     const edm::Event& iSetup) const;


  
  /// PFCandidates in which we'll look for pile up particles 
  edm::InputTag   inputTagPFCandidates_;
  
  /// verbose ?
  bool   verbose_;

};

#endif

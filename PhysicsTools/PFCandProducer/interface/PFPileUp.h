#ifndef PhysicsTools_PFCandProducer_PFPileUp_
#define PhysicsTools_PFCandProducer_PFPileUp_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

/**\class PFPileUp 
\brief Identifies pile-up candidates from a collection of PFCandidates, and 
produces the corresponding collection of PileUpCandidates.

\author Colin Bernet
\date   february 2008
*/




class PFPileUp : public edm::EDProducer {
 public:

  explicit PFPileUp(const edm::ParameterSet&);

  ~PFPileUp();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup & c);

 private:
  

  
  /// PFCandidates to be analyzed
  edm::InputTag   inputTagPFCandidates_;
  
  /// verbose ?
  bool   verbose_;

};

#endif

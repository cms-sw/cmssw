#ifndef PhysicsTools_JetMCAlgos_TauGenJetProducer_
#define PhysicsTools_JetMCAlgos_TauGenJetProducer_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"


/**\class TauGenJetProducer 
\brief builds a GenJet from the visible daughters of a tau

\author Colin Bernet
\date   february 2008
*/




class TauGenJetProducer : public edm::EDProducer {
 public:

  explicit TauGenJetProducer(const edm::ParameterSet&);

  ~TauGenJetProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup & c);

 private:
   
  /// Input PFCandidates
  edm::InputTag   inputTagGenParticles_;
  
  /// verbose ?
  bool   verbose_;

};

#endif

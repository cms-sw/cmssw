#ifndef PhysicsTools_TagAndProbe_ElectronMatchedCandidateProducer_h
#define PhysicsTools_TagAndProbe_ElectronMatchedCandidateProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations

class ElectronMatchedCandidateProducer : public edm::EDProducer 
{
 public:
  explicit ElectronMatchedCandidateProducer(const edm::ParameterSet&);
  ~ElectronMatchedCandidateProducer();

 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
      
  // ----------member data ---------------------------

  edm::InputTag electronCollection_;
  edm::InputTag scCollection_;
  double delRMatchingCut_;
};

#endif

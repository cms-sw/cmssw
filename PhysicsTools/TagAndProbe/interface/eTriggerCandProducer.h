#ifndef PhysicsTools_TagAndProbe_eTriggerCandProducer_h
#define PhysicsTools_TagAndProbe_eTriggerCandProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// forward declarations

class eTriggerCandProducer : public edm::EDProducer 
{
 public:
  explicit eTriggerCandProducer(const edm::ParameterSet&);
  ~eTriggerCandProducer();

 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
      
  edm::InputTag _inputProducer;
  edm::InputTag triggerEventTag_;
  edm::InputTag hltTag_;
  double delRMatchingCut_;


};

#endif

#ifndef PhysicsTools_TagAndProbe_TriggerMatchProducer_h
#define PhysicsTools_TagAndProbe_TriggerMatchProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include <string>

// forward declarations
template<class object>
class TriggerMatchProducer : public edm::EDProducer 
{
 public:
  explicit TriggerMatchProducer(const edm::ParameterSet&);
  ~TriggerMatchProducer();

 private:
  virtual void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;

  // ----------member data --------------------------
    
  edm::InputTag _inputProducer;
  edm::InputTag triggerEventTag_;
  edm::InputTag triggerResultsTag_;
  std::string hltTag_;
  double delRMatchingCut_;
  std::string filterName_;
  bool storeRefCollection_;
  //  bool isFilter_;
  //  bool printIndex_;
  bool changed_;
  HLTConfigProvider hltConfig_;
};
#include "DPGAnalysis/Skims/src/TriggerMatchProducer.icc"
#endif

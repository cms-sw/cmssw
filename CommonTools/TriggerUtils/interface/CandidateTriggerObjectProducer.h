#ifndef Configuration_Skimming_CandidateTriggerObjectProducer_h
#define Configuration_Skimming_CandidateTriggerObjectProducer_h

/** \class CandidateTriggerObjectProducer
 *
 *  
 *  This class creates a list of candidates based on the last accepted filter 
 *
 *  $Date: $
 *  $Revision:  $
 *
 *  \author Paolo Meridiani
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

//
// class declaration
//
class CandidateTriggerObjectProducer : public edm::EDProducer {
  
 public:
  explicit CandidateTriggerObjectProducer(const edm::ParameterSet&);
  ~CandidateTriggerObjectProducer();

 private:
  virtual void beginRun(edm::Run& iRun, edm::EventSetup const& iSetup);
  virtual void beginJob() {} ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {} ;

  /// module config parameters
  edm::InputTag triggerResultsTag_;
  edm::InputTag triggerEventTag_;
  std::string   triggerName_;

  /// additional class data memebers
  edm::Handle<edm::TriggerResults>   triggerResultsHandle_;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle_;
  HLTConfigProvider hltConfig_;

};
#endif

#ifndef Configuration_Skimming_CandidateTriggerObjectProducer_h
#define Configuration_Skimming_CandidateTriggerObjectProducer_h

/** \class CandidateTriggerObjectProducer
 *
 *
 *  This class creates a list of candidates based on the last accepted filter
 *
 *  \author Paolo Meridiani
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

//
// class declaration
//
class CandidateTriggerObjectProducer : public edm::stream::EDProducer<> {
public:
  explicit CandidateTriggerObjectProducer(const edm::ParameterSet&);
  ~CandidateTriggerObjectProducer() override;

private:
  void beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// module config parameters
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;
  const std::string processName_;
  const std::string triggerName_;

  /// additional class data memebers
  HLTPrescaleProvider hltPrescaleProvider_;
};
#endif

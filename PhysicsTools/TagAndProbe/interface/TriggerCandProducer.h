#ifndef PhysicsTools_TagAndProbe_TriggerCandProducer_h
#define PhysicsTools_TagAndProbe_TriggerCandProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "PhysicsTools/TagAndProbe/interface/TriggerCandProducer.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Common/interface/TriggerNames.h"

// forward declarations
template <class object>
class TriggerCandProducer : public edm::stream::EDProducer<> {
public:
  explicit TriggerCandProducer(const edm::ParameterSet&);
  ~TriggerCandProducer() override;

private:
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data --------------------------

  edm::InputTag _inputProducer;
  edm::EDGetTokenT<edm::View<object> > _inputProducerToken;
  edm::InputTag triggerEventTag_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;
  edm::InputTag triggerResultsTag_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  std::vector<edm::InputTag> hltTags_;
  edm::InputTag theRightHLTTag_;
  double delRMatchingCut_;
  double objEtMin_;
  double objEtaMax_;
  std::string filterName_;
  bool storeRefCollection_;
  bool antiSelect_;
  bool isTriggerOR_;
  bool isFilter_;
  bool noHltFiring_;
  bool printIndex_;
  bool changed_;
  HLTPrescaleProvider hltPrescaleProvider_;
  bool skipEvent_;
  bool matchUnprescaledTriggerOnly_;
};
#include "PhysicsTools/TagAndProbe/interface/TriggerCandProducer.icc"
#endif

#ifndef TriggerHelper_H
#define TriggerHelper_H

// -*- C++ -*-
//
// Package:    DQM/TrackerCommon
// Class:      TriggerHelper
//
//
/**
  \class    TriggerHelper TriggerHelper.h
  "DQM/TrackerCommon/interface/TriggerHelper.h" \brief    Provides a code based
  selection for trigger and DCS information in order to have no failing filters
  in the CMSSW path.

   [...]

  \author   Volker Adler
*/

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include <memory>

class AlCaRecoTriggerBits;
class AlCaRecoTriggerBitsRcd;
namespace edm {
  class ConsumesCollector;
  class ParameterSet;
}  // namespace edm

class TriggerHelper {
  // Utility classes
  edm::ESWatcher<AlCaRecoTriggerBitsRcd> *watchDB_;
  std::unique_ptr<L1GtUtils> l1Gt_;
  HLTConfigProvider hltConfig_;
  bool hltConfigInit_;
  // Configuration parameters
  bool andOr_;
  bool andOrDcs_;
  edm::InputTag dcsInputTag_;
  std::vector<int> dcsPartitions_;
  bool errorReplyDcs_;
  bool andOrGt_;
  edm::InputTag gtInputTag_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtInputToken_;
  std::string gtDBKey_;
  std::vector<std::string> gtLogicalExpressions_;
  bool errorReplyGt_;
  bool andOrL1_;
  std::string l1DBKey_;
  edm::ESGetToken<AlCaRecoTriggerBits, AlCaRecoTriggerBitsRcd> alcaRecotriggerBitsToken_;
  std::vector<std::string> l1LogicalExpressions_;
  bool errorReplyL1_;
  bool andOrHlt_;
  edm::InputTag hltInputTag_;
  std::string hltDBKey_;
  std::vector<std::string> hltLogicalExpressions_;
  bool errorReplyHlt_;
  // Switches
  bool on_;
  bool onDcs_;
  bool onGt_;
  bool onL1_;
  bool onHlt_;
  // Member constants
  const std::string configError_;

public:
  // Constructors must be called from the ED module's c'tor
  template <typename T>
  TriggerHelper(const edm::ParameterSet &config, edm::ConsumesCollector &&iC, T &module);

  template <typename T>
  TriggerHelper(const edm::ParameterSet &config, edm::ConsumesCollector &iC, T &module);

  ~TriggerHelper();

  // Public methods
  bool on() { return on_; }
  bool off() { return (!on_); }
  void initRun(const edm::Run &run,
               const edm::EventSetup &setup);  // To be called from beginRun() methods
  bool accept(const edm::Event &event,
              const edm::EventSetup &setup);  // To be called from analyze/filter() methods

private:
  // Private methods

  TriggerHelper(const edm::ParameterSet &config);

  // DCS
  bool acceptDcs(const edm::Event &event);
  bool acceptDcsPartition(const edm::Handle<DcsStatusCollection> &dcsStatus, int dcsPartition) const;

  // GT status bits
  bool acceptGt(const edm::Event &event);
  bool acceptGtLogicalExpression(const edm::Handle<L1GlobalTriggerReadoutRecord> &gtReadoutRecord,
                                 std::string gtLogicalExpression);

  // L1
  bool acceptL1(const edm::Event &event, const edm::EventSetup &setup);
  bool acceptL1LogicalExpression(const edm::Event &event, std::string l1LogicalExpression);

  // HLT
  bool acceptHlt(const edm::Event &event);
  bool acceptHltLogicalExpression(const edm::Handle<edm::TriggerResults> &hltTriggerResults,
                                  std::string hltLogicalExpression) const;

  // Algos
  std::vector<std::string> expressionsFromDB(const std::string &key, const edm::EventSetup &setup);
  bool negate(std::string &word) const;
};

template <typename T>
TriggerHelper::TriggerHelper(const edm::ParameterSet &config, edm::ConsumesCollector &&iC, T &module)
    : TriggerHelper(config, iC, module) {
  gtInputTag_ = config.getParameter<edm::InputTag>("gtInputTag");
  gtInputToken_ = iC.consumes<L1GlobalTriggerReadoutRecord>(gtInputTag_);
  alcaRecotriggerBitsToken_ = iC.esConsumes<AlCaRecoTriggerBits, AlCaRecoTriggerBitsRcd>();
}

template <typename T>
TriggerHelper::TriggerHelper(const edm::ParameterSet &config, edm::ConsumesCollector &iC, T &module)
    : TriggerHelper(config) {
  if (onL1_ && (!l1DBKey_.empty() || !l1LogicalExpressions_.empty())) {
    l1Gt_ = std::make_unique<L1GtUtils>(config, iC, false, module, L1GtUtils::UseEventSetupIn::Event);
  }
}

#endif

#ifndef GenericTriggerEventFlag_H
#define GenericTriggerEventFlag_H

// -*- C++ -*-
//
// Package:    CommonTools/TriggerUtils
// Class:      GenericTriggerEventFlag
//
// $Id: GenericTriggerEventFlag.h,v 1.5 2012/01/19 20:17:34 vadler Exp $
//
/**
  \class    GenericTriggerEventFlag GenericTriggerEventFlag.h "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
  \brief    Provides a code based selection for trigger and DCS information in order to have no failing filters in the CMSSW path.

   [...]

  \author   Volker Adler
  \version  $Id: GenericTriggerEventFlag.h,v 1.5 2012/01/19 20:17:34 vadler Exp $
*/

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <memory>
#include <string>

class GenericTriggerEventFlag {
  // Utility classes
  std::unique_ptr<edm::ESWatcher<AlCaRecoTriggerBitsRcd> > watchDB_;
  std::unique_ptr<L1GtUtils> l1Gt_;
  std::unique_ptr<l1t::L1TGlobalUtil> l1uGt_;
  HLTConfigProvider hltConfig_;
  bool hltConfigInit_;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> l1GtTriggerMenuToken_;
  edm::ESGetToken<AlCaRecoTriggerBits, AlCaRecoTriggerBitsRcd> alCaRecoTriggerBitsToken_;
  // Configuration parameters
  bool andOr_;
  std::string dbLabel_;
  unsigned verbose_;
  bool andOrDcs_;
  edm::InputTag dcsInputTag_;
  edm::EDGetTokenT<DcsStatusCollection> dcsInputToken_;
  std::vector<int> dcsPartitions_;
  bool errorReplyDcs_;
  bool andOrGt_;
  edm::InputTag gtInputTag_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtInputToken_;
  edm::InputTag gtEvmInputTag_;
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> gtEvmInputToken_;
  std::string gtDBKey_;
  std::vector<std::string> gtLogicalExpressions_;
  bool errorReplyGt_;
  bool andOrL1_;
  bool stage2_;
  bool l1BeforeMask_;
  std::string l1DBKey_;
  std::vector<std::string> l1LogicalExpressionsCache_;
  std::vector<std::string> l1LogicalExpressions_;
  bool errorReplyL1_;
  bool andOrHlt_;
  edm::InputTag hltInputTag_;
  edm::EDGetTokenT<edm::TriggerResults> hltInputToken_;
  std::string hltDBKey_;
  std::vector<std::string> hltLogicalExpressionsCache_;
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
  const std::string emptyKeyError_;

public:
  //so passing in the owning EDProducer is a pain for me (S. Harper)
  //and its only needed for legacy/stage1 L1 info which is mostly obsolete now
  //defined a new constructor which doesnt allow for the use of legacy/stage 1 L1, only stage2
  //so you no longer have to pass in the EDProducer
  //however I set things up such that its an error to try and configure the stage-1 L1 here
  //hence the extra private constructor
  //tldr: use these constructors, not the other two if unsure, if you get it wrong, there'll be an error
  //
  //The last constructor argument declares whether EventSetup
  //information is retrieved during beginRun, during the Event,
  //or during both. This is needed to declare which EventSetup
  //products are consumed. In the future, this will affect
  //when prefetching is done. Declare both and it will always
  //work, but there is some performance advantage to only
  //declaring the necessary one. With only a few exceptions,
  //existing clients call both initRun and accept (the two main
  //functions in this class getting EventSetup data), so EventSetup
  //objects might be retrieved in both periods. The argument defaults
  //to this. The function expressionsFromDB also gets data from
  //the EventSetup and is called by a few clients.
  GenericTriggerEventFlag(const edm::ParameterSet& config,
                          edm::ConsumesCollector&& iC,
                          l1t::UseEventSetupIn use = l1t::UseEventSetupIn::RunAndEvent)
      : GenericTriggerEventFlag(config, iC, use) {}
  GenericTriggerEventFlag(const edm::ParameterSet& config,
                          edm::ConsumesCollector& iC,
                          l1t::UseEventSetupIn use = l1t::UseEventSetupIn::RunAndEvent);

  // Constructors must be called from the ED module's c'tor
  template <typename T>
  GenericTriggerEventFlag(const edm::ParameterSet& config,
                          edm::ConsumesCollector&& iC,
                          T& module,
                          l1t::UseEventSetupIn use = l1t::UseEventSetupIn::RunAndEvent);

  template <typename T>
  GenericTriggerEventFlag(const edm::ParameterSet& config,
                          edm::ConsumesCollector& iC,
                          T& module,
                          l1t::UseEventSetupIn use = l1t::UseEventSetupIn::RunAndEvent);

  // Public methods
  bool on() { return on_; }
  bool off() { return (!on_); }
  void initRun(const edm::Run& run, const edm::EventSetup& setup);     // To be called from beginRun() methods
  bool accept(const edm::Event& event, const edm::EventSetup& setup);  // To be called from analyze/filter() methods

  bool allHLTPathsAreValid() const;

private:
  GenericTriggerEventFlag(const edm::ParameterSet& config, edm::ConsumesCollector& iC, bool stage1Valid);
  // Private methods

  // DCS
  bool acceptDcs(const edm::Event& event);
  bool acceptDcsPartition(const edm::Handle<DcsStatusCollection>& dcsStatus, int dcsPartition) const;

  // GT status bits
  bool acceptGt(const edm::Event& event);
  bool acceptGtLogicalExpression(const edm::Event& event, std::string gtLogicalExpression);

  // L1
  bool acceptL1(const edm::Event& event, const edm::EventSetup& setup);
  bool acceptL1LogicalExpression(const edm::Event& event,
                                 const edm::EventSetup& setup,
                                 std::string l1LogicalExpression);

  // HLT
  bool acceptHlt(const edm::Event& event);
  bool acceptHltLogicalExpression(const edm::Handle<edm::TriggerResults>& hltTriggerResults,
                                  std::string hltLogicalExpression) const;

  // Algos
  std::string expandLogicalExpression(const std::vector<std::string>& target,
                                      const std::string& expr,
                                      bool useAnd = false) const;
  bool negate(std::string& word) const;

public:
  // Methods for expert analysis

  std::string gtDBKey() { return gtDBKey_; }    // can be empty
  std::string l1DBKey() { return l1DBKey_; }    // can be empty
  std::string hltDBKey() { return hltDBKey_; }  // can be empty

  // Must be called only during beginRun
  std::vector<std::string> expressionsFromDB(const std::string& key, const edm::EventSetup& setup);
};

template <typename T>
GenericTriggerEventFlag::GenericTriggerEventFlag(const edm::ParameterSet& config,
                                                 edm::ConsumesCollector&& iC,
                                                 T& module,
                                                 l1t::UseEventSetupIn use)
    : GenericTriggerEventFlag(config, iC, module, use) {}

template <typename T>
GenericTriggerEventFlag::GenericTriggerEventFlag(const edm::ParameterSet& config,
                                                 edm::ConsumesCollector& iC,
                                                 T& module,
                                                 l1t::UseEventSetupIn use)
    : GenericTriggerEventFlag(config, iC, true) {
  if (on_ && config.exists("andOrL1")) {
    if (stage2_) {
      l1uGt_ = std::make_unique<l1t::L1TGlobalUtil>(config, iC, use);
    } else {
      L1GtUtils::UseEventSetupIn useL1GtUtilsIn = L1GtUtils::UseEventSetupIn::Run;
      if (use == l1t::UseEventSetupIn::RunAndEvent) {
        useL1GtUtilsIn = L1GtUtils::UseEventSetupIn::RunAndEvent;
      } else if (use == l1t::UseEventSetupIn::Event) {
        useL1GtUtilsIn = L1GtUtils::UseEventSetupIn::Event;
      }
      l1Gt_ = std::make_unique<L1GtUtils>(config, iC, false, module, useL1GtUtilsIn);
    }
  }
  //these pointers are already null so no need to reset them to a nullptr
  //if andOrL1 doesnt exist
}

#endif

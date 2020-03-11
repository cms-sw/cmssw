#ifndef HLTcore_HLTPrescaleProvider_h
#define HLTcore_HLTPrescaleProvider_h

/** \class HLTPrescaleProvider
 *
 *  
 *  This class provides access routines to get hold of the HLT Configuration
 *
 *
 *  \author Martin Grunewald
 *
 *  Originally the functions in here were in HLTConfigProvider.
 *  The functions that use L1GtUtils and get products from the
 *  Event were moved into this class in 2015 when the consumes
 *  function calls were added. W. David Dagenhart
 */

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
  class Run;
}  // namespace edm

class HLTPrescaleProvider {
public:
  template <typename T>
  HLTPrescaleProvider(edm::ParameterSet const& pset, edm::ConsumesCollector&& iC, T& module);

  template <typename T>
  HLTPrescaleProvider(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, T& module);

  /// Run-dependent initialisation (non-const method)
  ///   "init" return value indicates whether intitialisation has succeeded
  ///   "changed" parameter indicates whether the config has actually changed
  ///   This must be called at beginRun for most of the other functions in this class to succeed
  bool init(const edm::Run& iRun, const edm::EventSetup& iSetup, const std::string& processName, bool& changed);

  HLTConfigProvider const& hltConfigProvider() const { return hltConfigProvider_; }
  L1GtUtils const& l1GtUtils() const;
  l1t::L1TGlobalUtil const& l1tGlobalUtil() const;

  /// HLT prescale values via (L1) EventSetup
  /// current (default) prescale set index - to be taken from L1GtUtil via Event
  int prescaleSet(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  // negative == error

  /// combining the two methods above
  unsigned int prescaleValue(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& trigger);

  /// Combined L1T (pair.first) and HLT (pair.second) prescales per HLT path
  std::pair<int, int> prescaleValues(const edm::Event& iEvent,
                                     const edm::EventSetup& iSetup,
                                     const std::string& trigger);
  // any one negative => error in retrieving this (L1T or HLT) prescale

  // In case of a complex Boolean expression as L1 seed
  std::pair<std::vector<std::pair<std::string, int> >, int> prescaleValuesInDetail(const edm::Event& iEvent,
                                                                                   const edm::EventSetup& iSetup,
                                                                                   const std::string& trigger);
  // Event rejected by HLTPrescaler on ith HLT path?
  bool rejectedByHLTPrescaler(const edm::TriggerResults& triggerResults, unsigned int i) const;

private:
  void checkL1GtUtils() const;
  void checkL1TGlobalUtil() const;

  HLTConfigProvider hltConfigProvider_;
  std::unique_ptr<L1GtUtils> l1GtUtils_;
  std::unique_ptr<l1t::L1TGlobalUtil> l1tGlobalUtil_;
  unsigned char count_[5] = {0, 0, 0, 0, 0};
  bool inited_ = false;
};

template <typename T>
HLTPrescaleProvider::HLTPrescaleProvider(edm::ParameterSet const& pset, edm::ConsumesCollector&& iC, T& module)
    : HLTPrescaleProvider(pset, iC, module) {}

template <typename T>
HLTPrescaleProvider::HLTPrescaleProvider(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, T& module) {
  unsigned int stageL1Trigger = pset.getParameter<unsigned int>("stageL1Trigger");
  if (stageL1Trigger <= 1) {
    l1GtUtils_ = std::make_unique<L1GtUtils>(pset, iC, false, module, L1GtUtils::UseEventSetupIn::Run);
  } else {
    l1tGlobalUtil_ = std::make_unique<l1t::L1TGlobalUtil>(pset, iC, module, l1t::UseEventSetupIn::Run);
  }
}
#endif

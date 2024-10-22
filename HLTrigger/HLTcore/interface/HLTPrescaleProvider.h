#ifndef HLTrigger_HLTcore_HLTPrescaleProvider_h
#define HLTrigger_HLTcore_HLTPrescaleProvider_h

/** \class HLTPrescaleProvider
 *
 *  This class provides access routines to get hold of the HLT Configuration,
 *  as well as the prescales of Level-1 and High-Level triggers.
 *
 *  \author Martin Grunewald
 *
 *  Originally the functions in here were in HLTConfigProvider.
 *  The functions that use L1GtUtils and get products from the
 *  Event were moved into this class in 2015 when the consumes
 *  function calls were added. W. David Dagenhart
 */

#include "HLTrigger/HLTcore/interface/FractionalPrescale.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
  class Run;
  class ParameterSetDescription;
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
  template <typename T = unsigned int>
  T prescaleValue(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& trigger) {
    const int set(prescaleSet(iEvent, iSetup));
    //there is a template specialisation for unsigned in which returns +1 which
    //emulates old behaviour
    return set < 0 ? -1 : hltConfigProvider_.prescaleValue<T>(static_cast<unsigned int>(set), trigger);
  }

  /// Combined L1T (pair.first) and HLT (pair.second) prescales per HLT path
  template <typename TL1 = int, typename THLT = TL1>
  std::pair<TL1, THLT> prescaleValues(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      const std::string& trigger) {
    return {convertL1PS<TL1>(getL1PrescaleValue(iEvent, iSetup, trigger)),
            prescaleValue<THLT>(iEvent, iSetup, trigger)};
  }
  // any one negative => error in retrieving this (L1T or HLT) prescale

  // In case of a complex Boolean expression as L1 seed
  template <typename TL1 = int, typename THLT = TL1>
  std::pair<std::vector<std::pair<std::string, TL1>>, THLT> prescaleValuesInDetail(const edm::Event& iEvent,
                                                                                   const edm::EventSetup& iSetup,
                                                                                   const std::string& trigger) {
    std::pair<std::vector<std::pair<std::string, TL1>>, THLT> retval;
    for (auto& entry : getL1PrescaleValueInDetail(iEvent, iSetup, trigger)) {
      retval.first.emplace_back(std::move(entry.first), convertL1PS<TL1>(entry.second));
    }
    retval.second = prescaleValue<THLT>(iEvent, iSetup, trigger);
    return retval;
  }
  // Event rejected by HLTPrescaler on ith HLT path?
  bool rejectedByHLTPrescaler(const edm::TriggerResults& triggerResults, unsigned int i) const;
  static int l1PrescaleDenominator() { return kL1PrescaleDenominator_; }

  static void fillPSetDescription(edm::ParameterSetDescription& desc,
                                  unsigned int stageL1Trigger,
                                  edm::InputTag const& l1tAlgBlkInputTag,
                                  edm::InputTag const& l1tExtBlkInputTag,
                                  bool readPrescalesFromFile);

private:
  static constexpr const char* l1tGlobalDecisionKeyword_ = "L1GlobalDecision";

  void checkL1GtUtils() const;
  void checkL1TGlobalUtil() const;

  template <typename T>
  T convertL1PS(double val) const {
    static_assert(std::is_same_v<T, double> or std::is_same_v<T, FractionalPrescale>,
                  "\n\n\tPlease use convertL1PS<double> or convertL1PS<FractionalPrescale>"
                  " (other types for L1T prescales are not supported anymore by HLTPrescaleProvider)"
                  "\n\tconvertL1PS is used inside prescaleValues and prescaleValuesInDetail,"
                  " so it might be necessary to specify template arguments for those calls,"
                  "\n\te.g. prescaleValues<double, FractionalPrescale>"
                  " (the 1st argument applies to L1T prescales, the 2nd to HLT prescales)\n");
    return T(val);
  }

  double getL1PrescaleValue(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& trigger);

  std::vector<std::pair<std::string, double>> getL1PrescaleValueInDetail(const edm::Event& iEvent,
                                                                         const edm::EventSetup& iSetup,
                                                                         const std::string& trigger);

  static constexpr int kL1PrescaleDenominator_ = 100;

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

template <>
FractionalPrescale HLTPrescaleProvider::convertL1PS(double val) const;

#endif  // HLTrigger_HLTcore_HLTPrescaleProvider_h

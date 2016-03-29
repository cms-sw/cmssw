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

#include <string>
#include <utility>
#include <vector>

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
  class Run;
}

class HLTPrescaleProvider {

public:

  template <typename T>
  HLTPrescaleProvider(edm::ParameterSet const& pset,
                      edm::ConsumesCollector&& iC,
                      T& module);

  template <typename T>
  HLTPrescaleProvider(edm::ParameterSet const& pset,
                      edm::ConsumesCollector& iC,
                      T& module);

  /// Run-dependent initialisation (non-const method)
  ///   "init" return value indicates whether intitialisation has succeeded
  ///   "changed" parameter indicates whether the config has actually changed
  bool init(const edm::Run& iRun, const edm::EventSetup& iSetup,
	    const std::string& processName, bool& changed);

  HLTConfigProvider const& hltConfigProvider() const { return hltConfigProvider_; }
  L1GtUtils const& l1GtUtils() const { return l1GtUtils_; }

  /// HLT prescale values via (L1) EventSetup
  /// current (default) prescale set index - to be taken from L1GtUtil via Event
  int prescaleSet(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  // negative == error
  
  /// combining the two methods above
  unsigned int prescaleValue(const edm::Event& iEvent,
                             const edm::EventSetup& iSetup,
                             const std::string& trigger);

  /// Combined L1T (pair.first) and HLT (pair.second) prescales per HLT path
  std::pair<int,int> prescaleValues(const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    const std::string& trigger);
  // any one negative => error in retrieving this (L1T or HLT) prescale

  // In case of a complex Boolean expression as L1 seed
  std::pair<std::vector<std::pair<std::string,int> >,int> prescaleValuesInDetail(const edm::Event& iEvent,
                                                                                 const edm::EventSetup& iSetup,
                                                                                 const std::string& trigger);

 private:

  HLTConfigProvider hltConfigProvider_;
  L1GtUtils l1GtUtils_;
  unsigned char count_[5] = {0,0,0,0,0};

};

template <typename T>
HLTPrescaleProvider::HLTPrescaleProvider(edm::ParameterSet const& pset,
                                         edm::ConsumesCollector&& iC,
                                         T& module) :
  HLTPrescaleProvider(pset, iC, module) { }

template <typename T>
HLTPrescaleProvider::HLTPrescaleProvider(edm::ParameterSet const& pset,
                                         edm::ConsumesCollector& iC,
                                         T& module) :
  l1GtUtils_(pset, iC, false, module) { }
#endif

#ifndef HLTrigger_HLTcore_HLTEventAnalyzerAOD_h
#define HLTrigger_HLTcore_HLTEventAnalyzerAOD_h

/** \class HLTEventAnalyzerAOD
 *
 *  
 *  This class is an EDAnalyzer analyzing the combined HLT information for AOD
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class HLTEventAnalyzerAOD : public edm::stream::EDAnalyzer<> {
public:
  explicit HLTEventAnalyzerAOD(const edm::ParameterSet &);
  ~HLTEventAnalyzerAOD() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void endRun(edm::Run const &, edm::EventSetup const &) override {}

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void analyzeTrigger(const edm::Event &, const edm::EventSetup &, const std::string &triggerName);

private:
  using LOG = edm::LogVerbatim;

  static constexpr const char *logMsgType_ = "HLTEventAnalyzerAOD";

  /// module config parameters
  const std::string processName_;
  const std::string triggerName_;
  const edm::InputTag triggerResultsTag_;
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::InputTag triggerEventTag_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  /// additional class data members
  bool const verbose_;
  edm::Handle<edm::TriggerResults> triggerResultsHandle_;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle_;
  HLTPrescaleProvider hltPrescaleProvider_;
};

#endif  // HLTrigger_HLTcore_HLTEventAnalyzerAOD_h

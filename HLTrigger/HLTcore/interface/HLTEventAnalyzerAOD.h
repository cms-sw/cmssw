#ifndef HLTcore_HLTEventAnalyzerAOD_h
#define HLTcore_HLTEventAnalyzerAOD_h

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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class HLTEventAnalyzerAOD : public edm::stream::EDAnalyzer< > {
  
 public:
  explicit HLTEventAnalyzerAOD(const edm::ParameterSet&);
  virtual ~HLTEventAnalyzerAOD();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  virtual void endRun(edm::Run const &, edm::EventSetup const&) override;
  virtual void beginRun(edm::Run const &, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void analyzeTrigger(const edm::Event&, const edm::EventSetup&, const std::string& triggerName);

 private:

  /// module config parameters
  const std::string   processName_;
  const std::string   triggerName_;
  const edm::InputTag                           triggerResultsTag_;
  const edm::EDGetTokenT<edm::TriggerResults>   triggerResultsToken_;
  const edm::InputTag                           triggerEventTag_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  /// additional class data memebers
  edm::Handle<edm::TriggerResults>   triggerResultsHandle_;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle_;
  HLTConfigProvider hltConfig_;

};
#endif

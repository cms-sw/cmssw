#ifndef HLTcore_TriggerSummaryAnalyzerRAW_h
#define HLTcore_TriggerSummaryAnalyzerRAW_h

/** \class TriggerSummaryAnalyzerRAW
 *
 *  
 *  This class is an EDAnalyzer analyzing the HLT summary object for RAW
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class TriggerSummaryAnalyzerRAW : public edm::stream::EDAnalyzer<> {
  
 public:
  explicit TriggerSummaryAnalyzerRAW(const edm::ParameterSet&);
  virtual ~TriggerSummaryAnalyzerRAW();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  
 private:
  /// InputTag of TriggerEventWithRefs to analyze
  const edm::InputTag                                   inputTag_;
  const edm::EDGetTokenT<trigger::TriggerEventWithRefs> inputToken_;

};
#endif

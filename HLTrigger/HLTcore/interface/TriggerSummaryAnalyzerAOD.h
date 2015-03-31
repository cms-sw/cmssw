#ifndef HLTcore_TriggerSummaryAnalyzerAOD_h
#define HLTcore_TriggerSummaryAnalyzerAOD_h

/** \class TriggerSummaryAnalyzerAOD
 *
 *  
 *  This class is an EDAnalyzer analyzing the HLT summary object for AOD
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class TriggerSummaryAnalyzerAOD : public edm::stream::EDAnalyzer<> {
  
 public:
  explicit TriggerSummaryAnalyzerAOD(const edm::ParameterSet&);
  virtual ~TriggerSummaryAnalyzerAOD();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:
  /// InputTag of TriggerEvent to analyze
  const edm::InputTag                           inputTag_;
  const edm::EDGetTokenT<trigger::TriggerEvent> inputToken_;

};
#endif

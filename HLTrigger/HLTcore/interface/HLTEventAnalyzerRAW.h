#ifndef HLTcore_HLTEventAnalyzerRAW_h
#define HLTcore_HLTEventAnalyzerRAW_h

/** \class HLTEventAnalyzerRAW
 *
 *  
 *  This class is an EDAnalyzer analyzing the combined HLT information for RAW
 *
 *  $Date: 2008/09/06 12:01:51 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

//
// class declaration
//
class HLTEventAnalyzerRAW : public edm::EDAnalyzer {
  
 public:
  explicit HLTEventAnalyzerRAW(const edm::ParameterSet&);
  ~HLTEventAnalyzerRAW();

  virtual void beginRun(edm::Run const &, edm::EventSetup const&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void analyzeTrigger(const std::string& triggerName);

 private:

  /// module config parameters
  std::string   processName_;
  std::string   triggerName_;
  edm::InputTag triggerResultsTag_;
  edm::InputTag triggerEventWithRefsTag_;

  /// additional class data memebers
  edm::Handle<edm::TriggerResults>           triggerResultsHandle_;
  edm::Handle<trigger::TriggerEventWithRefs> triggerEventWithRefsHandle_;
  HLTConfigProvider hltConfig_;

};
#endif

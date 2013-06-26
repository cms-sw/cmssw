#ifndef HLTcore_TriggerSummaryAnalyzerRAW_h
#define HLTcore_TriggerSummaryAnalyzerRAW_h

/** \class TriggerSummaryAnalyzerRAW
 *
 *  
 *  This class is an EDAnalyzer analyzing the HLT summary object for RAW
 *
 *  $Date: 2008/04/11 18:09:11 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//
class TriggerSummaryAnalyzerRAW : public edm::EDAnalyzer {
  
 public:
  explicit TriggerSummaryAnalyzerRAW(const edm::ParameterSet&);
  ~TriggerSummaryAnalyzerRAW();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag inputTag_;

};
#endif

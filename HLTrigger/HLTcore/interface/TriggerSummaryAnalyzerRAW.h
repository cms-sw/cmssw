#ifndef HLTcore_TriggerSummaryAnalyzerRAW_h
#define HLTcore_TriggerSummaryAnalyzerRAW_h

/** \class TriggerSummaryAnalyzerRAW
 *
 *  
 *  This class is an EDAnalyzer analyzing the HLT summary object for RAW
 *
 *  $Date: 2007/12/06 08:27:31 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

//
// class declaration
//
class TriggerSummaryAnalyzerRAW : public edm::EDAnalyzer {
  
 public:
  explicit TriggerSummaryAnalyzerRAW(const edm::ParameterSet&);
  ~TriggerSummaryAnalyzerRAW();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag inputTag;

};
#endif

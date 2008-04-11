#ifndef HLTcore_TriggerSummaryAnalyzerAOD_h
#define HLTcore_TriggerSummaryAnalyzerAOD_h

/** \class TriggerSummaryAnalyzerAOD
 *
 *  
 *  This class is an EDAnalyzer analyzing the HLT summary object for AOD
 *
 *  $Date: 2008/01/12 16:53:56 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

//
// class declaration
//
class TriggerSummaryAnalyzerAOD : public edm::EDAnalyzer {
  
 public:
  explicit TriggerSummaryAnalyzerAOD(const edm::ParameterSet&);
  ~TriggerSummaryAnalyzerAOD();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  /// InputTag of TriggerEvent to analyze
  edm::InputTag inputTag;

};
#endif

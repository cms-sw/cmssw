#ifndef HLTcore_TriggerSummaryAnalyzerAOD_h
#define HLTcore_TriggerSummaryAnalyzerAOD_h

/** \class TriggerSummaryAnalyzerAOD
 *
 *  
 *  This class is an EDAnalyzer analyzing the HLT summary object for AOD
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
class TriggerSummaryAnalyzerAOD : public edm::EDAnalyzer {
  
 public:
  explicit TriggerSummaryAnalyzerAOD(const edm::ParameterSet&);
  ~TriggerSummaryAnalyzerAOD();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  /// InputTag of TriggerEvent to analyze
  edm::InputTag inputTag_;

};
#endif

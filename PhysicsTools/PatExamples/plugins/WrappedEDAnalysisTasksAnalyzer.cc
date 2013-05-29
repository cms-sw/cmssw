#include "FWCore/Framework/interface/MakerMacros.h"

/*
  This is an example of using the BasicMuonAnalyzer class to do simple analysis of muons both 
  in full framework and FWLite using the same c++ class. You can find the example to use the 
  code in FWLite in PhysicsTools/UtilAlgos/bin/FWLiteWithBasicAnalyzer.cc.
*/
#include "PhysicsTools/PatExamples/interface/AnalysisTasksAnalyzerBTag.h"
#include "PhysicsTools/PatExamples/interface/AnalysisTasksAnalyzerJEC.h"
#include "PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h"
typedef edm::AnalyzerWrapper<AnalysisTasksAnalyzerBTag> WrappedEDAnalysisTasksAnalyzerBTag;
DEFINE_FWK_MODULE(WrappedEDAnalysisTasksAnalyzerBTag);

typedef edm::AnalyzerWrapper<AnalysisTasksAnalyzerJEC> WrappedEDAnalysisTasksAnalyzerJEC;
DEFINE_FWK_MODULE(WrappedEDAnalysisTasksAnalyzerJEC);

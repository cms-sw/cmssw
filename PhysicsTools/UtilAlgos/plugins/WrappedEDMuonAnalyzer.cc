#include "FWCore/Framework/interface/MakerMacros.h"

/*
  This is an example of using the BasicMuonAnalyzer class to do simple analysis of muons both 
  in full framework and FWLite using the same c++ class. You can find the example to use the 
  code in FWLite in PhysicsTools/UtilAlgos/bin/FWLiteWithBasicAnalyzer.cc.
*/
#include "PhysicsTools/UtilAlgos/interface/BasicMuonAnalyzer.h"
#include "PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h"
typedef edm::AnalyzerWrapper<BasicMuonAnalyzer> WrappedEDMuonAnalyzer;
DEFINE_FWK_MODULE(WrappedEDMuonAnalyzer);

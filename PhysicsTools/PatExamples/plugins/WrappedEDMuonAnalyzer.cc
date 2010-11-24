#include "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"
#include "PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h"

typedef edm::AnalyzerWrapper<BasicMuonAnalyzer> WrappedEDMuonAnalyzer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(WrappedEDMuonAnalyzer);

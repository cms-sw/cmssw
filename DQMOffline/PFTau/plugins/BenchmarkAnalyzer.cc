#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

BenchmarkAnalyzer::BenchmarkAnalyzer(const edm::ParameterSet& iConfig)
{

  inputLabel_      = iConfig.getParameter<edm::InputTag>("InputCollection");
  outputFile_      = iConfig.getUntrackedParameter<std::string>("OutputFile");
  benchmarkLabel_  = iConfig.getParameter<std::string>("BenchmarkLabel"); 

  if (outputFile_.size() > 0)
    edm::LogInfo("OutputInfo") << " ParticleFLow Task histograms will be saved to '" << outputFile_.c_str()<< "'";
  else edm::LogInfo("OutputInfo") << " ParticleFlow Task histograms will NOT be saved";

}

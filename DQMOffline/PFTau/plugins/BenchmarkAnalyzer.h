#ifndef __DQMOffline_PFTau_BenchmarkAnalyzer__
#define __DQMOffline_PFTau_BenchmarkAnalyzer__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMOffline/PFTau/interface/Benchmark.h"

/// abtract base class for benchmark analyzers
class BenchmarkAnalyzer: public edm::EDAnalyzer {
public:

  BenchmarkAnalyzer();
  explicit BenchmarkAnalyzer(const edm::ParameterSet&);
  virtual ~BenchmarkAnalyzer() {}

  virtual void beginJob() = 0;

 protected:

  /// name of the output root file
  std::string outputFile_;
  
  /// input collection
  edm::InputTag inputLabel_;

  /// benchmark label
  std::string benchmarkLabel_;

};

#endif 

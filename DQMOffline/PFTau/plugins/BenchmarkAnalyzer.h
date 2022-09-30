#ifndef __DQMOffline_PFTau_BenchmarkAnalyzer__
#define __DQMOffline_PFTau_BenchmarkAnalyzer__

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

/// abtract base class for benchmark analyzers
class BenchmarkAnalyzer : public DQMEDAnalyzer {
public:
  /// Constructors
  BenchmarkAnalyzer();
  explicit BenchmarkAnalyzer(const edm::ParameterSet &);

  /// Destructor
  ~BenchmarkAnalyzer() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

protected:
  /// name of the output root file
  std::string outputFile_;

  /// input collection
  edm::InputTag inputLabel_;

  /// benchmark label
  std::string benchmarkLabel_;

  std::string eventInfoFolder_;
  std::string subsystemname_;
};

#endif

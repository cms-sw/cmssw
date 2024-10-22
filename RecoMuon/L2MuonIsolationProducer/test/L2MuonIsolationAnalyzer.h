#ifndef RecoMuon_L2MuonIsolationProducer_L2MuonIsolationAnalyzer_H
#define RecoMuon_L2MuonIsolationProducer_L2MuonIsolationAnalyzer_H

/** \class L2MuonIsolationAnalyzer
 *  Analyzer of HLT L2 muon isolation performance
 *
 *  \author J. Alcaraz
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TFile;
class TH1F;
class TH2F;

class L2MuonIsolationAnalyzer : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  L2MuonIsolationAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  ~L2MuonIsolationAnalyzer() override;

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  void beginJob() override;
  void endJob() override;

private:
  void Puts(const char* fmt, ...);

  // Isolation label
  edm::InputTag theIsolationLabel;

  // Cone and Et sets to be tested
  std::vector<double> theConeCases;
  double theEtMin;
  double theEtMax;
  unsigned int theEtBins;

  // Reference isolation cuts
  muonisolation::Cuts theCuts;

  // Root output file
  std::string theRootFileName;
  TFile* theRootFile;

  // Text output file
  std::string theTxtFileName;
  FILE* theTxtFile;

  // Histograms
  TH1F* hEtSum;
  TH1F* hEffVsCone;
  TH1F* hEffVsEt;
  std::vector<TH1F*> hEffVsEtArray;

  // Counters and vectors
  unsigned int numberOfEvents;
  unsigned int numberOfMuons;
};
#endif

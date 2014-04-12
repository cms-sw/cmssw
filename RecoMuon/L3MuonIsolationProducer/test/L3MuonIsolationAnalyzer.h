#ifndef RecoMuon_L3MuonIsolationProducer_L3MuonIsolationAnalyzer_H
#define RecoMuon_L3MuonIsolationProducer_L3MuonIsolationAnalyzer_H

/** \class L3MuonIsolationAnalyzer
 *  Analyzer of HLT L3 muon isolation performance
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class TH2F;

class L3MuonIsolationAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  L3MuonIsolationAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~L3MuonIsolationAnalyzer();

  // Operations
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;

private:
  void Puts(const char* fmt, ...);

  // Isolation label
  edm::InputTag theIsolationLabel;

  // Cone and Pt sets to be tested
  std::vector<double> theConeCases;
  double thePtMin;
  double thePtMax;
  unsigned int thePtBins;

  // Reference isolation cuts
  muonisolation::Cuts theCuts;

  // Root output file
  std::string theRootFileName;
  TFile* theRootFile;

  // Text output file
  std::string theTxtFileName;
  FILE* theTxtFile;

  // Histograms
  TH1F *hPtSum;
  TH1F *hEffVsCone;
  TH1F *hEffVsPt;
  std::vector<TH1F*> hEffVsPtArray;

  // Counters and vectors
  unsigned int numberOfEvents;
  unsigned int numberOfMuons;
  
};
#endif


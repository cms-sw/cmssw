#ifndef RecoMuon_L2MuonIsolationProducer_L2MuonIsolationAnalyzer_H
#define RecoMuon_L2MuonIsolationProducer_L2MuonIsolationAnalyzer_H

/** \class L2MuonIsolationAnalyzer
 *  Analyzer of HLT L2 muon isolation performance
 *
 *  \author J. Alcaraz
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

class L2MuonIsolationAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  L2MuonIsolationAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~L2MuonIsolationAnalyzer();

  // Operations
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;

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
  TH1F *hEtSum;
  TH1F *hEffVsCone;
  TH1F *hEffVsEt;
  std::vector<TH1F*> hEffVsEtArray;

  // Counters and vectors
  unsigned int numberOfEvents;
  unsigned int numberOfMuons;
  
};
#endif


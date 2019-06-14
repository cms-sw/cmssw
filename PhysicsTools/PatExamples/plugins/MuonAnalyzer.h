#ifndef TutorialAtDESY_MuonAnalyzer_H
#define TutorialAtDESY_MuonAnalyzer_H

/** \class ExampleMuonAnalyzer
 *  Analyzer of the muon objects
 *
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TH1I;
class TH1F;
class TH2F;

class ExampleMuonAnalyzer : public edm::EDAnalyzer {
public:
  /// Constructor
  ExampleMuonAnalyzer(const edm::ParameterSet &pset);

  /// Destructor
  ~ExampleMuonAnalyzer() override;

  // Operations

  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

  void beginJob() override;
  void endJob() override;

protected:
private:
  edm::EDGetTokenT<pat::MuonCollection> theMuonToken;

  // Histograms
  TH1I *hNMuons;
  TH1F *hPtRec;
  TH2F *hPtReso;
  TH1F *hEHcal;

  TH1I *hMuonType;
  TH1F *hPtSTATKDiff;

  // ID
  TH1F *hMuCaloCompatibility;
  TH1F *hMuSegCompatibility;
  TH1I *hChamberMatched;
  TH1I *hMuIdAlgo;

  // Isolation
  TH1F *hMuIso03SumPt;
  TH1F *hMuIso03CaloComb;

  TH1F *h4MuInvMass;
};
#endif

#ifndef TutorialAtDESY_MuonAnalyzer_H
#define TutorialAtDESY_MuonAnalyzer_H

/** \class ExampleMuonAnalyzer
 *  Analyzer of the muon objects
 *
 *  $Date: 2009/11/06 10:06:22 $
 *  $Revision: 1.4 $
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TH1I;
class TH1F;
class TH2F;

class ExampleMuonAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  ExampleMuonAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~ExampleMuonAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;
protected:

private:
  std::string theMuonLabel;

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


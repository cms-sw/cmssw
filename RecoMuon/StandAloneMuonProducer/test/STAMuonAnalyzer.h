#ifndef RecoMuon_StandAloneMuonProducer_STAMuonAnalyzer_H
#define RecoMuon_StandAloneMuonProducer_STAMuonAnalyzer_H

/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2009/10/31 05:19:45 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class TH2F;

class STAMuonAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  STAMuonAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~STAMuonAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;
protected:

private:
  std::string theRootFileName;
  TFile* theFile;

  std::string theSTAMuonLabel;
  std::string theSeedCollectionLabel;

  // Histograms
  TH1F *hPtRec;
  TH1F *hPtSim; 
  TH1F *hPres;
  TH1F *h1_Pres;
  TH1F *hPTDiff;
  TH1F *hPTDiff2;
  TH2F *hPTDiffvsEta;
  TH2F *hPTDiffvsPhi;

  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;

  std::string theDataType;
  
};
#endif


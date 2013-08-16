#ifndef RecoMuon_GlobalMuonProducer_GLBMuonAnalyzer_H
#define RecoMuon_GlobalMuonProducer_GLBMuonAnalyzer_H

/** \class GLBMuonAnalyzer
 *  Analyzer of the Global muon tracks
 *
 *  \author R. Bellan  - INFN Torino       <riccardo.bellan@cern.ch>
 *  \author A. Everett - Purdue University <adam.everett@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

//#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TFile;
class TH1F;
class TH2F;

class GLBMuonAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  GLBMuonAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~GLBMuonAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;
protected:

private:
  edm::InputTag theGLBMuonLabel;
  
  std::string theRootFileName;
  TFile* theFile;
  
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


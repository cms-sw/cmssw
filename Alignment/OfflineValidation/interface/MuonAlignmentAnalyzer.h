#ifndef Alignment_OfflineValidation_MuonAlignmentAnalyzer_H
#define Alignment_OfflineValidation_MuonAlignmentAnalyzer_H

/** \class MuonAlignmentANalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2007/07/16 18:03:36 $
 *  $Revision: 1.4 $
 *  \author J. Fernandez - IFCA (CSIC-UC) <Javier.Fernandez@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}

class TFile;
class TH1F;
class TH2F;

using namespace std;
using namespace edm;

class MuonAlignmentAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  MuonAlignmentAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~MuonAlignmentAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob(const edm::EventSetup& eventSetup) ;
  virtual void endJob() ;
protected:

private:
  std::string theRootFileName;
  TFile* theFile;

// InputTags
edm::InputTag theGLBMuonTag; 
edm::InputTag theSTAMuonTag; 

// Collections needed
  edm::InputTag theRecHits4DTagDT;
  edm::InputTag theRecHits2DTagCSC;

// To switch between real data and MC
  std::string theDataType;

  bool doSAplots,doGBplots,doResplots;

  // Histograms
  TH1F	*hNmuonsGB;
  TH1F	*hNmuonsSA;
  TH1F	*hNmuonsSim;
  TH1F *hGBChi2;
  TH1F *hSAChi2;
  TH1F *hGBInvM;
  TH1F *hSAInvM;
  TH1F *hSimInvM;
  TH1F *hGBInvM_Barrel;
  TH1F *hSAInvM_Barrel;
  TH1F *hSimInvM_Barrel;
  TH1F *hSAPTRec;
  TH1F *hGBPTRec;
  TH1F *hPTSim; 
  TH1F *hSAPTres;
  TH1F *hSAinvPTres;
  TH1F *hGBPTres;
  TH1F *hGBinvPTres;
  TH1F *hSAPTDiff;
  TH1F *hGBPTDiff;
  TH2F *hSAPTDiffvsEta;
  TH2F *hSAPTDiffvsPhi;
  TH2F *hGBPTDiffvsEta;
  TH2F *hGBPTDiffvsPhi;
  TH2F *hGBPTvsEta;
  TH2F *hGBPTvsPhi;
  TH2F *hSAPTvsEta;
  TH2F *hSAPTvsPhi;
  TH2F *hSimPTvsEta;
  TH2F *hSimPTvsPhi;
  TH2F	*hGBinvPTvsEta;
  TH2F	*hGBinvPTvsPhi;
  TH2F	*hSAinvPTvsEta;
  TH2F	*hSAinvPTvsPhi;

  // Vector of chambers Residuals
  std::vector<TH1F *> unitsRPhi;
  std::vector<TH1F *> unitsPhi;
  std::vector<TH1F *> unitsZ;
  TH1F *hResidualRPhiDT; 
  TH1F *hResidualPhiDT; 
  TH1F *hResidualZDT; 
  TH1F *hResidualRPhiCSC; 
  TH1F *hResidualPhiCSC; 
  TH1F *hResidualZCSC; 
  TH1F *hResidualRPhiDT_W[5];
  TH1F *hResidualPhiDT_W[5];
  TH1F *hResidualZDT_W[5];	
  TH1F *hResidualRPhiCSC_ME[18];
  TH1F *hResidualPhiCSC_ME[18];
  TH1F *hResidualZCSC_ME[18];
  TH1F *hResidualRPhiDT_MB[20];
  TH1F *hResidualPhiDT_MB[20];
  TH1F *hResidualZDT_MB[20];

  std::vector<long> detectorCollection;  


  // Counters
  int numberOfSimTracks;
  int numberOfGBRecTracks;
  int numberOfSARecTracks;
  int numberOfHits;
  
};
#endif


#ifndef Alignment_OfflineValidation_MuonAlignmentAnalyzer_H
#define Alignment_OfflineValidation_MuonAlignmentAnalyzer_H

/** \class MuonAlignmentANalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2007/07/19 17:53:05 $
 *  $Revision: 1.5 $
 *  \author J. Fernandez - IFCA (CSIC-UC) <Javier.Fernandez@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include <vector>

namespace edm {
  class ParameterSet;
  class EventSetup;
  class InputTag;
}

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
  edm::Service<TFileService> fs;

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

  //# muons per event
  TH1F	*hNmuonsGB;
  TH1F	*hNmuonsSA;
  TH1F	*hNmuonsSim;

  // Chi2 of Track
  TH1F *hGBChi2;
  TH1F *hSAChi2;

  // Invariant mass for dimuons
  TH1F *hGBInvM;
  TH1F *hSAInvM;
  TH1F *hSimInvM;
  // Invariant Mass distributions in Barrel (eta<1.04) region
  TH1F *hGBInvM_Barrel;
  TH1F *hSAInvM_Barrel;
  TH1F *hSimInvM_Barrel;

  // pT 
  TH1F *hSAPTRec;
  TH1F *hGBPTRec;
  TH1F *hPTSim; 
  TH2F *hGBPTvsEta;
  TH2F *hGBPTvsPhi;
  TH2F *hSAPTvsEta;
  TH2F *hSAPTvsPhi;
  TH2F *hSimPTvsEta;
  TH2F *hSimPTvsPhi;

  // pT resolution
  TH1F *hSAPTres;
  TH1F *hSAinvPTres;
  TH1F *hGBPTres;
  TH1F *hGBinvPTres;
  //pT rec - pT gen
  TH1F *hSAPTDiff;
  TH1F *hGBPTDiff;
  TH2F *hSAPTDiffvsEta;
  TH2F *hSAPTDiffvsPhi;
  TH2F *hGBPTDiffvsEta;
  TH2F *hGBPTDiffvsPhi;
  TH2F	*hGBinvPTvsEta;
  TH2F	*hGBinvPTvsPhi;
  TH2F	*hSAinvPTvsEta;
  TH2F	*hSAinvPTvsPhi;



  // Vector of chambers Residuals
  std::vector<TH1F *> unitsRPhi;
  std::vector<TH1F *> unitsPhi;
  std::vector<TH1F *> unitsZ;

  // DT & CSC Residuals
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

  ESHandle<MagneticField> theMGField;

  Propagator * thePropagator;

  // Counters
  int numberOfSimTracks;
  int numberOfGBRecTracks;
  int numberOfSARecTracks;
  int numberOfHits;
  
};
#endif


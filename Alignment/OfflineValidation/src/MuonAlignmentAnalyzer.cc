/** \class MuonAlignmentAnalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2007/07/16 18:03:37 $
 *  $Revision: 1.4 $
 *  \author J. Fernandez - IFCA (CSIC-UC) <Javier.Fernandez@cern.ch>
 */

#include "Alignment/OfflineValidation/interface/MuonAlignmentAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLorentzVector.h"

using namespace std;
using namespace edm;

/// Constructor
MuonAlignmentAnalyzer::MuonAlignmentAnalyzer(const ParameterSet& pset){
  theSTAMuonTag = pset.getParameter<edm::InputTag>("StandAloneTrackCollectionTag");
  theGLBMuonTag = pset.getParameter<edm::InputTag>("GlobalMuonTrackCollectionTag");

  theRecHits4DTagDT = pset.getParameter<edm::InputTag>("RecHits4DDTCollectionTag");
  theRecHits2DTagCSC = pset.getParameter<edm::InputTag>("RecHits2DCSCCollectionTag");
  
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  theDataType = pset.getUntrackedParameter<string>("DataType");

  doSAplots = pset.getUntrackedParameter<bool>("doSAplots");
  doGBplots = pset.getUntrackedParameter<bool>("doGBplots");
  doResplots = pset.getUntrackedParameter<bool>("doResplots");
 
  if(theDataType != "RealData" && theDataType != "SimData")
    edm::LogError("MuonAlignmentAnalyzer") << "Error in Data Type!!"<<endl;

  numberOfSimTracks=0;
  numberOfSARecTracks=0;
  numberOfGBRecTracks=0;
  numberOfHits=0;
}

/// Destructor
MuonAlignmentAnalyzer::~MuonAlignmentAnalyzer(){
}

void MuonAlignmentAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

// Define and book histograms for SA and GB muon quantities/objects

//# muons per event
  hNmuonsGB = new TH1F("GBNmuons","Nmuons",10,0,10);
  hNmuonsSA = new TH1F("SANmuons","Nmuons",10,0,10);
  hNmuonsSim = new TH1F("SimNmuons","Nmuons",10,0,10);

// ############     pt    ####################
  hGBPTRec = new TH1F("GBpTRec","p_{T}^{rec}",300,0,300);
  hSAPTRec = new TH1F("SApTRec","p_{T}^{rec}",300,0,300);
  hPTSim = new TH1F("pTSim","p_{T}^{gen} ",300,0,300);

  hGBPTvsEta = new TH2F("GBPTvsEta","p_{T}^{rec} VS #eta",100,-2.5,2.5,300,0,300);
  hGBPTvsPhi = new TH2F("GBPTvsPhi","p_{T}^{rec} VS #phi",100,-6,6,300,0,300);

  hSAPTvsEta = new TH2F("SAPTvsEta","p_{T}^{rec} VS #eta",100,-2.5,2.5,300,0,300);
  hSAPTvsPhi = new TH2F("SAPTvsPhi","p_{T}^{rec} VS #phi",100,-6,6,300,0,300);

  hSimPTvsEta = new TH2F("SimPTvsEta","p_{T}^{gen} VS #eta",100,-2.5,2.5,300,0,300);
  hSimPTvsPhi = new TH2F("SimPTvsPhi","p_{T}^{gen} VS #phi",100,-6,6,300,0,300);

//pT rec - pT gen
  hSAPTDiff = new TH1F("SApTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);
  hGBPTDiff = new TH1F("GBpTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);

  hSAPTDiffvsEta = new TH2F("SAPTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hSAPTDiffvsPhi = new TH2F("SAPTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,250,-120,120);

  hGBPTDiffvsEta = new TH2F("GBPTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hGBPTDiffvsPhi = new TH2F("GBPTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-6,6,250,-120,120);

// pT resolution
  hSAPTres = new TH1F("SApTRes","pT Resolution",100,-2,2);
  hSAinvPTres = new TH1F("SAinvPTRes","1/pT Resolution",100,-2,2);
  hGBPTres = new TH1F("GBpTRes","pT Resolution",100,-2,2);
  hGBinvPTres = new TH1F("GBinvPTRes","1/pT Resolution",100,-2,2);
  
  hGBinvPTvsEta = new TH2F("GBinvPTvsEta","1/p_{T}^{rec} VS #eta",100,-2.5,2.5,100,-2,2);
  hGBinvPTvsPhi = new TH2F("GBinvPTvsPhi","1/p_{T}^{rec} VS #phi",100,-6,6,100,-2,2);

  hSAinvPTvsEta = new TH2F("SAinvPTvsEta","1/p_{T}^{rec} VS #eta",100,-2.5,2.5,100,-2,2);
  hSAinvPTvsPhi = new TH2F("SAinvPTvsPhi","1/p_{T}^{rec} VS #phi",100,-6,6,100,-2,2);


// #######Invariant mass #############
// invariant mass for dimuons
  hGBInvM = new TH1F("GBInvM","M_{inv}^{rec}",200,0,200);
  hSAInvM = new TH1F("SAInvM","M_{inv}^{rec}",200,0,200);
  hSimInvM = new TH1F("SimInvM","M_{inv}^{gen} ",200,0,200);

// Invariant Mass distributions in Barrel (eta<1.04) region
  hGBInvM_Barrel = new TH1F("GBInvM_Barrel","M_{inv}^{rec}",200,0,200);
  hSAInvM_Barrel = new TH1F("SAInvM_Barrel","M_{inv}^{rec}",200,0,200);
  hSimInvM_Barrel = new TH1F("SimInvM_Barrel","M_{inv}^{rec}",200,0,200);

 // Chi2 of Track
  hGBChi2 = new TH1F("GBChi2","Chi2",200,0,200);
  hSAChi2 = new TH1F("SAChi2","Chi2",200,0,200);

// All DT and CSC chambers 
  hResidualRPhiDT = new TH1F("hResidualRPhiDT","hResidualRPhiDT",200,-10,10);
  hResidualPhiDT = new TH1F("hResidualPhiDT","hResidualPhiDT",100,-1,1);
  hResidualZDT = new TH1F("hResidualZDT","hResidualZDT",200,-10,10);
  hResidualRPhiCSC = new TH1F("hResidualRPhiCSC","hResidualRPhiCSC",200,-10,10);
  hResidualPhiCSC = new TH1F("hResidualPhiCSC","hResidualPhiCSC",100,-1,1);
  hResidualZCSC = new TH1F("hResidualZCSC","hResidualZCSC",200,-10,10);

// DT Wheels
  hResidualRPhiDT_W[0]=new TH1F("hResidualRPhiDT_W-2","hResidualRPhiDT_W-2",200,-10,10);
  hResidualPhiDT_W[0]=new TH1F("hResidualPhiDT_W-2","hResidualPhiDT_W-2",200,-1,1);
  hResidualZDT_W[0] = new TH1F("hResidualZDT_W-2","hResidualZDT_W-2",200,-10,10);
  hResidualRPhiDT_W[1]=new TH1F("hResidualRPhiDT_W-1","hResidualRPhiDT_W-1",200,-10,10);
  hResidualPhiDT_W[1]=new TH1F("hResidualPhiDT_W-1","hResidualPhiDT_W-1",200,-1,1);
  hResidualZDT_W[1] = new TH1F("hResidualZDT_W-1","hResidualZDT_W-1",200,-10,10);
  hResidualRPhiDT_W[2]=new TH1F("hResidualRPhiDT_W0","hResidualRPhiDT_W0",200,-10,10);
  hResidualPhiDT_W[2]=new TH1F("hResidualPhiDT_W0","hResidualPhiDT_W0",200,-1,1);
  hResidualZDT_W[2] = new TH1F("hResidualZDT_W0","hResidualZDT_W0",200,-10,10);
  hResidualRPhiDT_W[3]=new TH1F("hResidualRPhiDT_W1","hResidualRPhiDT_W1",200,-10,10);
  hResidualPhiDT_W[3]=new TH1F("hResidualPhiDT_W1","hResidualPhiDT_W1",200,-1,1);
  hResidualZDT_W[3] = new TH1F("hResidualZDT_W1","hResidualZDT_W1",200,-10,10);
  hResidualRPhiDT_W[4]=new TH1F("hResidualRPhiDT_W2","hResidualRPhiDT_W2",200,-10,10);
  hResidualPhiDT_W[4]=new TH1F("hResidualPhiDT_W2","hResidualPhiDT_W2",200,-1,1);
  hResidualZDT_W[4] = new TH1F("hResidualZDT_W2","hResidualZDT_W2",200,-10,10);

// DT Stations
  hResidualRPhiDT_MB[0]=new TH1F("hResidualRPhiDT_MB-2/1","hResidualRPhiDT_MB-2/1",200,-10,10);
  hResidualPhiDT_MB[0]=new TH1F("hResidualPhiDT_MB-2/1","hResidualPhiDT_MB-2/1",200,-1,1);
  hResidualZDT_MB[0] = new TH1F("hResidualZDT_MB-2/1","hResidualZDT_MB-2/1",200,-10,10);
  hResidualRPhiDT_MB[1]=new TH1F("hResidualRPhiDT_MB-2/2","hResidualRPhiDT_MB-2/2",200,-10,10);
  hResidualPhiDT_MB[1]=new TH1F("hResidualPhiDT_MB-2/2","hResidualPhiDT_MB-2/2",200,-1,1);
  hResidualZDT_MB[1] = new TH1F("hResidualZDT_MB-2/2","hResidualZDT_MB-2/2",200,-10,10);
  hResidualRPhiDT_MB[2]=new TH1F("hResidualRPhiDT_MB-2/3","hResidualRPhiDT_MB-2/3",200,-10,10);
  hResidualPhiDT_MB[2]=new TH1F("hResidualPhiDT_MB-2/3","hResidualPhiDT_MB-2/3",200,-1,1);
  hResidualZDT_MB[2] = new TH1F("hResidualZDT_MB-2/3","hResidualZDT_MB-2/3",200,-10,10);
  hResidualRPhiDT_MB[3]=new TH1F("hResidualRPhiDT_MB-2/4","hResidualRPhiDT_MB-2/4",200,-10,10);
  hResidualPhiDT_MB[3]=new TH1F("hResidualPhiDT_MB-2/4","hResidualPhiDT_MB-2/4",200,-1,1);
  hResidualZDT_MB[3] = new TH1F("hResidualZDT_MB-2/4","hResidualZDT_MB-2/4",200,-10,10);
  hResidualRPhiDT_MB[4]=new TH1F("hResidualRPhiDT_MB-1/1","hResidualRPhiDT_MB-1/1",200,-10,10);
  hResidualPhiDT_MB[4]=new TH1F("hResidualPhiDT_MB-1/1","hResidualPhiDT_MB-1/1",200,-1,1);
  hResidualZDT_MB[4] = new TH1F("hResidualZDT_MB-1/1","hResidualZDT_MB-1/1",200,-10,10);
  hResidualRPhiDT_MB[5]=new TH1F("hResidualRPhiDT_MB-1/2","hResidualRPhiDT_MB-1/2",200,-10,10);
  hResidualPhiDT_MB[5]=new TH1F("hResidualPhiDT_MB-1/2","hResidualPhiDT_MB-1/2",200,-1,1);
  hResidualZDT_MB[5] = new TH1F("hResidualZDT_MB-1/2","hResidualZDT_MB-1/2",200,-10,10);
  hResidualRPhiDT_MB[6]=new TH1F("hResidualRPhiDT_MB-1/3","hResidualRPhiDT_MB-1/3",200,-10,10);
  hResidualPhiDT_MB[6]=new TH1F("hResidualPhiDT_MB-1/3","hResidualPhiDT_MB-1/3",200,-1,1);
  hResidualZDT_MB[6] = new TH1F("hResidualZDT_MB-1/3","hResidualZDT_MB-1/3",200,-10,10);
  hResidualRPhiDT_MB[7]=new TH1F("hResidualRPhiDT_MB-1/4","hResidualRPhiDT_MB-1/4",200,-10,10);
  hResidualPhiDT_MB[7]=new TH1F("hResidualPhiDT_MB-1/4","hResidualPhiDT_MB-1/4",200,-1,1);
  hResidualZDT_MB[7] = new TH1F("hResidualZDT_MB-1/4","hResidualZDT_MB-1/4",200,-10,10);
  hResidualRPhiDT_MB[8]=new TH1F("hResidualRPhiDT_MB0/1","hResidualRPhiDT_MB0/1",200,-10,10);
  hResidualPhiDT_MB[8]=new TH1F("hResidualPhiDT_MB0/1","hResidualPhiDT_MB0/1",200,-1,1);
  hResidualZDT_MB[8] = new TH1F("hResidualZDT_MB0/1","hResidualZDT_MB0/1",200,-10,10);
  hResidualRPhiDT_MB[9]=new TH1F("hResidualRPhiDT_MB0/2","hResidualRPhiDT_MB0/2",200,-10,10);
  hResidualPhiDT_MB[9]=new TH1F("hResidualPhiDT_MB0/2","hResidualPhiDT_MB0/2",200,-1,1);
  hResidualZDT_MB[9] = new TH1F("hResidualZDT_MB0/2","hResidualZDT_MB0/2",200,-10,10);
  hResidualRPhiDT_MB[10]=new TH1F("hResidualRPhiDT_MB0/3","hResidualRPhiDT_MB0/3",200,-10,10);
  hResidualPhiDT_MB[10]=new TH1F("hResidualPhiDT_MB0/3","hResidualPhiDT_MB0/3",200,-1,1);
  hResidualZDT_MB[10] = new TH1F("hResidualZDT_MB0/3","hResidualZDT_MB0/3",200,-10,10);
  hResidualRPhiDT_MB[11]=new TH1F("hResidualRPhiDT_MB0/4","hResidualRPhiDT_MB0/4",200,-10,10);
  hResidualPhiDT_MB[11]=new TH1F("hResidualPhiDT_MB0/4","hResidualPhiDT_MB0/4",200,-1,1);
  hResidualZDT_MB[11] = new TH1F("hResidualZDT_MB0/4","hResidualZDT_MB0/4",200,-10,10);
  hResidualRPhiDT_MB[12]=new TH1F("hResidualRPhiDT_MB1/1","hResidualRPhiDT_MB1/1",200,-10,10);
  hResidualPhiDT_MB[12]=new TH1F("hResidualPhiDT_MB1/1","hResidualPhiDT_MB1/1",200,-1,1);
  hResidualZDT_MB[12] = new TH1F("hResidualZDT_MB1/1","hResidualZDT_MB1/1",200,-10,10);
  hResidualRPhiDT_MB[13]=new TH1F("hResidualRPhiDT_MB1/2","hResidualRPhiDT_MB1/2",200,-10,10);
  hResidualPhiDT_MB[13]=new TH1F("hResidualPhiDT_MB1/2","hResidualPhiDT_MB1/2",200,-1,1);
  hResidualZDT_MB[13] = new TH1F("hResidualZDT_MB1/2","hResidualZDT_MB1/2",200,-10,10);
  hResidualRPhiDT_MB[14]=new TH1F("hResidualRPhiDT_MB1/3","hResidualRPhiDT_MB1/3",200,-10,10);
  hResidualPhiDT_MB[14]=new TH1F("hResidualPhiDT_MB1/3","hResidualPhiDT_MB1/3",200,-1,1);
  hResidualZDT_MB[14] = new TH1F("hResidualZDT_MB1/3","hResidualZDT_MB1/3",200,-10,10);
  hResidualRPhiDT_MB[15]=new TH1F("hResidualRPhiDT_MB1/4","hResidualRPhiDT_MB1/4",200,-10,10);
  hResidualPhiDT_MB[15]=new TH1F("hResidualPhiDT_MB1/4","hResidualPhiDT_MB1/4",200,-1,1);
  hResidualZDT_MB[15] = new TH1F("hResidualZDT_MB1/4","hResidualZDT_MB1/4",200,-10,10);
  hResidualRPhiDT_MB[16]=new TH1F("hResidualRPhiDT_MB2/1","hResidualRPhiDT_MB2/1",200,-10,10);
  hResidualPhiDT_MB[16]=new TH1F("hResidualPhiDT_MB2/1","hResidualPhiDT_MB2/1",200,-1,1);
  hResidualZDT_MB[16] = new TH1F("hResidualZDT_MB2/1","hResidualZDT_MB2/1",200,-10,10);
  hResidualRPhiDT_MB[17]=new TH1F("hResidualRPhiDT_MB2/2","hResidualRPhiDT_MB2/2",200,-10,10);
  hResidualPhiDT_MB[17]=new TH1F("hResidualPhiDT_MB2/2","hResidualPhiDT_MB2/2",200,-1,1);
  hResidualZDT_MB[17] = new TH1F("hResidualZDT_MB2/2","hResidualZDT_MB2/2",200,-10,10);
  hResidualRPhiDT_MB[18]=new TH1F("hResidualRPhiDT_MB2/3","hResidualRPhiDT_MB2/3",200,-10,10);
  hResidualPhiDT_MB[18]=new TH1F("hResidualPhiDT_MB2/3","hResidualPhiDT_MB2/3",200,-1,1);
  hResidualZDT_MB[18] = new TH1F("hResidualZDT_MB2/3","hResidualZDT_MB2/3",200,-10,10);
  hResidualRPhiDT_MB[19]=new TH1F("hResidualRPhiDT_MB2/4","hResidualRPhiDT_MB2/4",200,-10,10);
  hResidualPhiDT_MB[19]=new TH1F("hResidualPhiDT_MB2/4","hResidualPhiDT_MB2/4",200,-1,1);
  hResidualZDT_MB[19] = new TH1F("hResidualZDT_MB2/4","hResidualZDT_MB2/4",200,-10,10);


// CSC Stations
  hResidualRPhiCSC_ME[0]=new TH1F("hResidualRPhiCSC_ME-4/1","hResidualRPhiCSC_ME-4/1",200,-10,10);
  hResidualPhiCSC_ME[0]=new TH1F("hResidualPhiCSC_ME-4/1","hResidualPhiCSC_ME-4/1",200,-1,1);
  hResidualZCSC_ME[0] = new TH1F("hResidualZCSC_ME-4/1","hResidualZCSC_ME-4/1",200,-10,10);
  hResidualRPhiCSC_ME[1]=new TH1F("hResidualRPhiCSC_ME-4/2","hResidualRPhiCSC_ME-4/2",200,-10,10);
  hResidualPhiCSC_ME[1]=new TH1F("hResidualPhiCSC_ME-4/2","hResidualPhiCSC_ME-4/2",200,-1,1);
  hResidualZCSC_ME[1] = new TH1F("hResidualZCSC_ME-4/2","hResidualZCSC_ME-4/2",200,-10,10);
  hResidualRPhiCSC_ME[2]=new TH1F("hResidualRPhiCSC_ME-3/1","hResidualRPhiCSC_ME-3/1",200,-10,10);
  hResidualPhiCSC_ME[2]=new TH1F("hResidualPhiCSC_ME-3/1","hResidualPhiCSC_ME-3/1",200,-1,1);
  hResidualZCSC_ME[2] = new TH1F("hResidualZCSC_ME-3/1","hResidualZCSC_ME-3/1",200,-10,10);
  hResidualRPhiCSC_ME[3]=new TH1F("hResidualRPhiCSC_ME-3/2","hResidualRPhiCSC_ME-3/2",200,-10,10);
  hResidualPhiCSC_ME[3]=new TH1F("hResidualPhiCSC_ME-3/2","hResidualPhiCSC_ME-3/2",200,-1,1);
  hResidualZCSC_ME[3] = new TH1F("hResidualZCSC_ME-3/2","hResidualZCSC_ME-3/2",200,-10,10);
  hResidualRPhiCSC_ME[4]=new TH1F("hResidualRPhiCSC_ME-2/1","hResidualRPhiCSC_ME-2/1",200,-10,10);
  hResidualPhiCSC_ME[4]=new TH1F("hResidualPhiCSC_ME-2/1","hResidualPhiCSC_ME-2/1",200,-1,1);
  hResidualZCSC_ME[4] = new TH1F("hResidualZCSC_ME-2/1","hResidualZCSC_ME-2/1",200,-10,10);
  hResidualRPhiCSC_ME[5]=new TH1F("hResidualRPhiCSC_ME-2/2","hResidualRPhiCSC_ME-2/2",200,-10,10);
  hResidualPhiCSC_ME[5]=new TH1F("hResidualPhiCSC_ME-2/2","hResidualPhiCSC_ME-2/2",200,-1,1);
  hResidualZCSC_ME[5] = new TH1F("hResidualZCSC_ME-2/2","hResidualZCSC_ME-2/2",200,-10,10);
  hResidualRPhiCSC_ME[6]=new TH1F("hResidualRPhiCSC_ME-1/1","hResidualRPhiCSC_ME-1/1",200,-10,10);
  hResidualPhiCSC_ME[6]=new TH1F("hResidualPhiCSC_ME-1/1","hResidualPhiCSC_ME-1/1",200,-1,1);
  hResidualZCSC_ME[6] = new TH1F("hResidualZCSC_ME-1/1","hResidualZCSC_ME-1/1",200,-10,10);
  hResidualRPhiCSC_ME[7]=new TH1F("hResidualRPhiCSC_ME-1/2","hResidualRPhiCSC_ME-1/2",200,-10,10);
  hResidualPhiCSC_ME[7]=new TH1F("hResidualPhiCSC_ME-1/2","hResidualPhiCSC_ME-1/2",200,-1,1);
  hResidualZCSC_ME[7] = new TH1F("hResidualZCSC_ME-1/2","hResidualZCSC_ME-1/2",200,-10,10);
  hResidualRPhiCSC_ME[8]=new TH1F("hResidualRPhiCSC_ME-1/3","hResidualRPhiCSC_ME-1/3",200,-10,10);
  hResidualPhiCSC_ME[8]=new TH1F("hResidualPhiCSC_ME-1/3","hResidualPhiCSC_ME-1/3",200,-1,1);
  hResidualZCSC_ME[8] = new TH1F("hResidualZCSC_ME-1/3","hResidualZCSC_ME-1/3",200,-10,10);
  hResidualRPhiCSC_ME[9]=new TH1F("hResidualRPhiCSC_ME1/1","hResidualRPhiCSC_ME1/1",200,-10,10);
  hResidualPhiCSC_ME[9]=new TH1F("hResidualPhiCSC_ME1/1","hResidualPhiCSC_ME1/1",100,-1,1);
  hResidualZCSC_ME[9] = new TH1F("hResidualZCSC_ME1/1","hResidualZCSC_ME1/1",200,-10,10);
  hResidualRPhiCSC_ME[10]=new TH1F("hResidualRPhiCSC_ME1/2","hResidualRPhiCSC_ME1/2",200,-10,10);
  hResidualPhiCSC_ME[10]=new TH1F("hResidualPhiCSC_ME1/2","hResidualPhiCSC_ME1/2",200,-1,1);
  hResidualZCSC_ME[10] = new TH1F("hResidualZCSC_ME1/2","hResidualZCSC_ME1/2",200,-10,10);
  hResidualRPhiCSC_ME[11]=new TH1F("hResidualRPhiCSC_ME1/3","hResidualRPhiCSC_ME1/3",200,-10,10);
  hResidualPhiCSC_ME[11]=new TH1F("hResidualPhiCSC_ME1/3","hResidualPhiCSC_ME1/3",200,-1,1);
  hResidualZCSC_ME[11] = new TH1F("hResidualZCSC_ME1/3","hResidualZCSC_ME1/3",200,-10,10);
  hResidualRPhiCSC_ME[12]=new TH1F("hResidualRPhiCSC_ME2/1","hResidualRPhiCSC_ME2/1",200,-10,10);
  hResidualPhiCSC_ME[12]=new TH1F("hResidualPhiCSC_ME2/1","hResidualPhiCSC_ME2/1",200,-1,1);
  hResidualZCSC_ME[12] = new TH1F("hResidualZCSC_ME2/1","hResidualZCSC_ME2/1",200,-10,10);
  hResidualRPhiCSC_ME[13]=new TH1F("hResidualRPhiCSC_ME2/2","hResidualRPhiCSC_ME2/2",200,-10,10);
  hResidualPhiCSC_ME[13]=new TH1F("hResidualPhiCSC_ME2/2","hResidualPhiCSC_ME2/2",200,-1,1);
  hResidualZCSC_ME[13] = new TH1F("hResidualZCSC_ME2/2","hResidualZCSC_ME2/2",200,-10,10);
  hResidualRPhiCSC_ME[14]=new TH1F("hResidualRPhiCSC_ME3/1","hResidualRPhiCSC_ME3/1",200,-10,10);
  hResidualPhiCSC_ME[14]=new TH1F("hResidualPhiCSC_ME3/1","hResidualPhiCSC_ME3/1",200,-1,1);
  hResidualZCSC_ME[14] = new TH1F("hResidualZCSC_ME3/1","hResidualZCSC_ME3/1",200,-10,10);
  hResidualRPhiCSC_ME[15]=new TH1F("hResidualRPhiCSC_ME3/2","hResidualRPhiCSC_ME3/2",200,-10,10);
  hResidualPhiCSC_ME[15]=new TH1F("hResidualPhiCSC_ME3/2","hResidualPhiCSC_ME3/2",200,-1,1);
  hResidualZCSC_ME[15] = new TH1F("hResidualZCSC_ME3/2","hResidualZCSC_ME3/2",200,-10,10);
  hResidualRPhiCSC_ME[16]=new TH1F("hResidualRPhiCSC_ME4/1","hResidualRPhiCSC_ME4/1",200,-10,10);
  hResidualPhiCSC_ME[16]=new TH1F("hResidualPhiCSC_ME4/1","hResidualPhiCSC_ME4/1",200,-1,1);
  hResidualZCSC_ME[16] = new TH1F("hResidualZCSC_ME4/1","hResidualZCSC_ME4/1",200,-10,10);
  hResidualRPhiCSC_ME[17]=new TH1F("hResidualRPhiCSC_ME4/2","hResidualRPhiCSC_ME4/2",200,-10,10);
  hResidualPhiCSC_ME[17]=new TH1F("hResidualPhiCSC_ME4/2","hResidualPhiCSC_ME4/2",200,-1,1);
  hResidualZCSC_ME[17] = new TH1F("hResidualZCSC_ME4/2","hResidualZCSC_ME4/2",200,-10,10);


}

void MuonAlignmentAnalyzer::endJob(){
  // Write the histos to file
  theFile->cd();

    edm::LogInfo("MuonAlignmentAnalyzer") << "----------------- " << endl << endl;

  if(theDataType == "SimData"){
    edm::LogInfo("MuonAlignmentAnalyzer") << "Number of Sim tracks: " << numberOfSimTracks << endl << endl;
  hNmuonsSim->Write();
  hPTSim->Write();
  hSimInvM->Write();
  hSimInvM_Barrel->Write();
  hSimPTvsEta->Write();
  hSimPTvsPhi->Write();
  }

  if(doSAplots){
    edm::LogInfo("MuonAlignmentAnalyzer") << "Number of SA Reco tracks: " << numberOfSARecTracks << endl << endl;
  hNmuonsSA->Write();
  if(theDataType == "SimData"){
  hSAPTres->Write();
  hSAinvPTres->Write();
  hSAinvPTvsEta->Write();
  hSAinvPTvsPhi->Write();
  hSAPTDiff->Write();
  hSAPTDiffvsEta->Write();
  hSAPTDiffvsPhi->Write();
	}
  hSAPTvsEta->Write();
  hSAPTvsPhi->Write();
  hSAPTRec->Write();
  hSAInvM->Write();
  hSAInvM_Barrel->Write();
  hSAChi2->Write();
	}

  if(doGBplots){
  edm::LogInfo("MuonAlignmentAnalyzer") << "Number of GB Reco tracks: " << numberOfGBRecTracks << endl << endl;
  hNmuonsGB->Write();
  if(theDataType == "SimData"){
  hGBPTres->Write();
  hGBPTDiff->Write();
  hGBPTDiffvsEta->Write();
  hGBPTDiffvsPhi->Write();
  hGBinvPTres->Write();
  hGBinvPTvsEta->Write();
  hGBinvPTvsPhi->Write();
	}
  hGBPTRec->Write();
  hGBPTvsEta->Write();
  hGBPTvsPhi->Write();
  hGBInvM->Write();
  hGBInvM_Barrel->Write();
  hGBChi2->Write();
   }

  if(doResplots){
  edm::LogInfo("MuonAlignmentAnalyzer") << "Number of Hits considered for residuals: " << numberOfHits << endl << endl;

    for(std::vector<TH1F *>::iterator myIt = unitsRPhi.begin(); myIt != unitsRPhi.end(); myIt++) {
      (*myIt)->Write();
    } 
    for(std::vector<TH1F *>::iterator myIt = unitsPhi.begin(); myIt != unitsPhi.end(); myIt++) {
      (*myIt)->Write();
    } 
    for(std::vector<TH1F *>::iterator myIt = unitsZ.begin(); myIt != unitsZ.end(); myIt++) {
      (*myIt)->Write();
    } 

  hResidualRPhiDT->Write(); 
  hResidualPhiDT->Write(); 
  hResidualZDT->Write(); 
  hResidualRPhiCSC->Write(); 
  hResidualPhiCSC->Write(); 
  hResidualZCSC->Write(); 
  
  for (int j=0; j<20; j++){

	if(j<5){
  hResidualRPhiDT_W[j]->Write();
  hResidualPhiDT_W[j]->Write();
  hResidualZDT_W[j]->Write();	
	}
	if(j<18){
  hResidualRPhiCSC_ME[j]->Write();
  hResidualPhiCSC_ME[j]->Write();
  hResidualZCSC_ME[j]->Write();
  	}
  hResidualRPhiDT_MB[j]->Write();
  hResidualPhiDT_MB[j]->Write();
  hResidualZDT_MB[j]->Write();
  }
}

  theFile->Close();
}
 

void MuonAlignmentAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  
  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  
  GlobalVector p1,p2;
  std::vector< double > simPar[3] ; //pt,eta,phi
  int i=0;
  

// ######### if data= MC, do Simulation Plots#####
  if(theDataType == "SimData"){
  double simEta=0;
  double simPt=0;
  double simPhi=0;

  // Get the SimTrack collection from the event
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
  

    SimTrackContainer::const_iterator simTrack;

	i=0;
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
      if (abs((*simTrack).type()) == 13) {
	i++;
	simPt=(*simTrack).momentum().perp();
	simEta=(*simTrack).momentum().eta();
	simPhi=(*simTrack).momentum().phi();
	numberOfSimTracks++;
	hPTSim->Fill(simPt);
	hSimPTvsEta->Fill(simEta,simPt);
	hSimPTvsPhi->Fill(simPhi,simPt);

	simPar[0].push_back(simPt);
	simPar[1].push_back(simEta);
	simPar[2].push_back(simPhi);

	
//	Save the muon pair
        if(i==1)  p1=GlobalVector((*simTrack).momentum().x(),(*simTrack).momentum().y(),(*simTrack).momentum().z());
    	if(i==2)  p2=GlobalVector((*simTrack).momentum().x(),(*simTrack).momentum().y(),(*simTrack).momentum().z());
     }    
    }
	hNmuonsSim->Fill(i);

  if(i==2){
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hSimInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04) hSimInvM_Barrel->Fill(Minv);
  }

  } //simData
  
  
// ############ Stand Alone Muon plots ###############  
  if(doSAplots){

  double SArecPt=0.;
  double SAeta=0.;
  double SAphi=0.;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonTag, staTracks);
  numberOfSARecTracks += staTracks->size();

  reco::TrackCollection::const_iterator staTrack;

  i=0;
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    i++;
    
    
    SArecPt = (*staTrack).pt();
    hSAPTRec->Fill(SArecPt);
    SAeta = (*staTrack).eta();
    SAphi = (*staTrack).phi();
    hSAChi2->Fill((*staTrack).chi2());

// save the muon pair
    if(i==1)  p1=GlobalVector((*staTrack).momentum().x(),(*staTrack).momentum().y(),(*staTrack).momentum().z());
    if(i==2)  p2=GlobalVector((*staTrack).momentum().x(),(*staTrack).momentum().y(),(*staTrack).momentum().z());

   
    if(SArecPt && theDataType == "SimData"){  
	
     for(unsigned int  iSim = 0; iSim <simPar[0].size(); iSim++) {
     double simPt=simPar[0][iSim];
     if(sqrt((SAeta-simPar[1][iSim])*(SAeta-simPar[1][iSim])+(SAphi-simPar[2][iSim])*(SAphi-simPar[2][iSim]))<0.3){
      hSAPTres->Fill( (SArecPt-simPt)/simPt);

      hSAPTDiff->Fill(SArecPt-simPt);

      hSAPTDiffvsEta->Fill(SAeta,SArecPt-simPt);
      hSAPTDiffvsPhi->Fill(SAphi,SArecPt-simPt);

      hSAinvPTres->Fill( ( 1/SArecPt - 1/simPt)/ (1/simPt));

      hSAinvPTvsEta->Fill(SAeta,( 1/SArecPt - 1/simPt)/ (1/simPt));
      hSAinvPTvsPhi->Fill(SAphi,( 1/SArecPt - 1/simPt)/ (1/simPt));
	}}
    }

      hSAPTvsEta->Fill(SAeta,SArecPt);
      hSAPTvsPhi->Fill(SAphi,SArecPt);
}

	hNmuonsSA->Fill(i);

  if(i==2){
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hSAInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04) hSAInvM_Barrel->Fill(Minv);
  }

  }//end doSAplots



// ############### Global Muons plots ##########

  if(doGBplots){  
  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> glbTracks;
  event.getByLabel(theGLBMuonTag, glbTracks);
  numberOfGBRecTracks += glbTracks->size();

  double GBrecPt = 0; 
  double GBeta = 0;
  double GBphi = 0;

  reco::TrackCollection::const_iterator glbTrack;
	i=0;

  for (glbTrack = glbTracks->begin(); glbTrack != glbTracks->end(); ++glbTrack){
    i++;

    GBrecPt = (*glbTrack).pt(); 
    GBeta = (*glbTrack).eta();
    GBphi = (*glbTrack).phi();
    
// save the muon pair
    if(i==1)  p1=GlobalVector((*glbTrack).momentum().x(),(*glbTrack).momentum().y(),(*glbTrack).momentum().z());
    if(i==2)  p2=GlobalVector((*glbTrack).momentum().x(),(*glbTrack).momentum().y(),(*glbTrack).momentum().z());

  
    hGBPTRec->Fill(GBrecPt);
    hGBChi2->Fill((*glbTrack).chi2());
   

    if(GBrecPt && theDataType == "SimData"){  
     for(unsigned int  iSim = 0; iSim <simPar[0].size(); iSim++) {
     double simPt=simPar[0][iSim];
     if(sqrt((GBeta-simPar[1][iSim])*(GBeta-simPar[1][iSim])+(GBphi-simPar[2][iSim])*(GBphi-simPar[2][iSim]))<0.3){
	
      hGBPTres->Fill( (GBrecPt-simPt)/simPt);

      hGBPTDiff->Fill(GBrecPt-simPt);

      hGBPTDiffvsEta->Fill(GBeta,GBrecPt-simPt);
      hGBPTDiffvsPhi->Fill(GBphi,GBrecPt-simPt);

      hGBinvPTres->Fill( ( 1/GBrecPt - 1/simPt)/ (1/simPt));
      hGBinvPTvsEta->Fill(GBeta,( 1/GBrecPt - 1/simPt)/ (1/simPt));
      hGBinvPTvsPhi->Fill(GBphi,( 1/GBrecPt - 1/simPt)/ (1/simPt));
	}}
    } 


      hGBPTvsEta->Fill(GBeta,GBrecPt);
      hGBPTvsPhi->Fill(GBphi,GBrecPt);

  }

	hNmuonsGB->Fill(i);

  if(i==2){
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hGBInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04)   hGBInvM_Barrel->Fill(Minv);
  }
} //end doGBplots


// ############    Residual plots ###################

 if(doResplots){

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonTag, staTracks);

  // Get the 4D DTSegments
  edm::Handle<DTRecSegment4DCollection> all4DSegmentsDT;
  event.getByLabel(theRecHits4DTagDT, all4DSegmentsDT);
  DTRecSegment4DCollection::const_iterator segmentDT;

  // Get the 2D CSCSegments
  edm::Handle<CSCSegmentCollection> all2DSegmentsCSC;
  event.getByLabel(theRecHits2DTagCSC, all2DSegmentsCSC);
  CSCSegmentCollection::const_iterator segmentCSC;
  
  //Create the propagator
    Propagator *thePropagator = new SteppingHelixPropagator(&*theMGField, alongMomentum);

   //Vectors used to perform the matching between Segments and hits from Track
  std::vector< std::vector<int> > indexCollectionDT;
  std::vector< std::vector<int> > indexCollectionCSC;
   

  reco::TrackCollection::const_iterator staTrack;
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){

	int countPoints   = 0;

    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    
    std::vector<int> positionDT;
    std::vector<int> positionCSC;
    std::vector<TrackingRecHit *> my4DTrack;
    
    //Loop over the hits of the track
    for(int counter  = 0; counter != staTrack->recHitsSize()-1; ++counter) {
  
      TrackingRecHitRef myRef = staTrack->recHit(counter);
      const TrackingRecHit *rechit = myRef.get();
      const GeomDet* geomDet = theTrackingGeometry->idToDet(rechit->geographicalId());
      
	//It's a DT Hit
      if(geomDet->subDetector() == GeomDetEnumerators::DT) {

	//Take the layer associated to this hit
	DTLayerId myLayer(rechit->geographicalId().rawId());
	
	int NumberOfDTSegment = 0;
	//Loop over segments
	for(segmentDT = all4DSegmentsDT->begin(); segmentDT != all4DSegmentsDT->end(); ++segmentDT) {
	  
	  //By default the chamber associated to this Segment is new
	  bool isNewChamber = true;

	  //Loop over segments already included in the vector of segments in the actual track
	  for(std::vector<int>::iterator positionIt = positionDT.begin(); positionIt != positionDT.end(); ++positionIt) {

	    //If this segment has been used before isNewChamber = false
	    if(NumberOfDTSegment == *positionIt) isNewChamber = false;
	  }

	  //Loop over vectors of segments associated to previous tracks
	  for(std::vector<std::vector<int> >::iterator collect = indexCollectionDT.begin(); collect != indexCollectionDT.end(); ++collect) {

	    //Loop over segments associated to a track
	    for(std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end(); positionIt++) {
	      
	      //If this segment was used in a previos track then isNewChamber = false
	      if(NumberOfDTSegment == *positionIt) isNewChamber = false;
	    }
	  }
	  
	  //If the chamber is new
	  if(isNewChamber) {
	    
	    DTChamberId myChamber((*segmentDT).geographicalId().rawId());
	    //If the layer of the hit belongs to the chamber of the 4D Segment
	    if(myLayer.wheel() == myChamber.wheel() && myLayer.station() == myChamber.station() && myLayer.sector() == myChamber.sector()) {
	      
	      //push position of the segment and tracking rechit
	      positionDT.push_back(NumberOfDTSegment);
	      my4DTrack.push_back((TrackingRecHit *) &(*segmentDT));
	    }
	  } //new Chamber if
	  
	  NumberOfDTSegment++;
	} // DTsegment Loop

      } //was a DT if
      
	//In case is a CSC
      else if (geomDet->subDetector() == GeomDetEnumerators::CSC) {
	
      //Take the layer associated to this hit
      CSCDetId myLayer(rechit->geographicalId().rawId());
	
      int NumberOfCSCSegment = 0;
      //Loop over 2Dsegments
      for(segmentCSC = all2DSegmentsCSC->begin(); segmentCSC != all2DSegmentsCSC->end(); segmentCSC++) {

      //By default the chamber associated to the segment is new
        bool isNewChamber = true;

      //Loop over segments in the current track
        for(std::vector<int>::iterator positionIt = positionCSC.begin(); positionIt != positionCSC.end(); positionIt++) {
	    
      //If this segment has been used then newchamber = false
          if(NumberOfCSCSegment == *positionIt) isNewChamber = false;
        }
      //Loop over vectors of segments in previous tracks
        for(std::vector<std::vector<int> >::iterator collect = indexCollectionCSC.begin(); collect != indexCollectionCSC.end(); ++collect) {
      //Loop over segments in a track
      for(std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end(); positionIt++) {
      //If the segment was used in a previous track isNewChamber = false
            if(NumberOfCSCSegment == *positionIt) isNewChamber = false;
          }
        }
      //  //If the chamber is new
        if(isNewChamber) {
      CSCDetId myChamber((*segmentCSC).geographicalId().rawId());
      //If the chambers are the same
          if(myLayer.chamberId() == myChamber.chamberId()) {
      //push
      positionCSC.push_back(NumberOfCSCSegment);
      my4DTrack.push_back((TrackingRecHit *) &(*segmentCSC));
          }
        }
        NumberOfCSCSegment++;
      } //CSC segment loop
    } //was a CSC if
    

// start propagation
    TrajectoryStateOnSurface innerTSOS = track.impactPointState();

    //If the state is valid
    if(innerTSOS.isValid()) {

      //Loop over Associated segments
      for(std::vector<TrackingRecHit *>::iterator rechit = my4DTrack.begin(); rechit != my4DTrack.end(); ++rechit) {
	
	const GeomDet* geomDet = theTrackingGeometry->idToDet((*rechit)->geographicalId());
	//This try block is to catch some exceptions given by the propagator
	try {
 
	  TrajectoryStateOnSurface destiny = thePropagator->propagate(*(innerTSOS.freeState()), geomDet->surface());
          if(!destiny.isValid()) continue;

	double residualRPhi,residualPhi,residualZ;

  residualRPhi = (*rechit)->localPosition().x() - destiny.localPosition().x();
  residualZ = geomDet->toGlobal((*rechit)->localPosition()).z() - destiny.freeState()->position().z();
  residualPhi = atan2(geomDet->toGlobal(((RecSegment *)(*rechit))->localDirection()).y(), 
  geomDet->toGlobal(((RecSegment*)(*rechit))->localDirection()).x())-atan2(destiny.globalDirection().y(), destiny.globalDirection().x());


          const long rawId= (*rechit)->geographicalId().rawId();
	  int position = -1;
	  bool newDetector = true; 
	  //Loop over the AligmentDetectorCollection to see if the detector is new and requires a new entry
	  for(std::vector<long>::iterator myIds = detectorCollection.begin(); myIds != detectorCollection.end(); myIds++) {
		++position;
	    //If matches newDetector = false
	    if(*myIds == rawId) {
	      newDetector = false;
	      break;
	    }
	  }

            DetId myDet(rawId);
            int det = myDet.subdetId();
            int wheel=0,station=0,sector=0;
            int endcap=0,ring=0,chamber=0;

            //If it's a DT
            if(det == 1) {
              DTChamberId myChamber(rawId);
              wheel=myChamber.wheel();
              station = myChamber.station();
              sector=myChamber.sector();
            } else if (det==2){
              CSCDetId myChamber(rawId);
              endcap= myChamber.endcap();
              station = myChamber.station();
                if(endcap==2) station = -station;
              ring = myChamber.ring();
              chamber=myChamber.chamber();

            }

	  if(newDetector) {
	    
	    //Create an RawIdDetector, fill it and push it into the collection 
	    detectorCollection.push_back(rawId);

	    //This piece of code calculates the range of the residuals
	    double range = 0.5;
	    switch(abs(station)) {
	    case 1:
	      range = 0.5;
	      break;
	    case 2:
	      range = 1.0;
	      break;
	    case 3:
	      range = 3.0;
	      break;
	    case 4:
	      range = 10.0;
	      break;
	    default:
	      break;
	    }

	//create histograms

	    char nameOfHistoRPhi[50];
	    char nameOfHistoPhi[50];
	    char nameOfHistoZ[50];

	
	    if(det==1){ // DT
	    sprintf(nameOfHistoRPhi, "ResidualRPhi_W%ldMB%1dS%1d",wheel,station,sector );
	    sprintf(nameOfHistoPhi, "ResidualPhi_W%ldMB%1dS%1d",wheel,station,sector);
	    sprintf(nameOfHistoZ, "ResidualZ_W%ldMB%1dS%1d",wheel,station,sector);
	    hResidualRPhiDT->Fill(residualRPhi);
	    hResidualPhiDT->Fill(residualPhi);
	    hResidualZDT->Fill(residualZ);			

		int index = wheel+2;
		hResidualRPhiDT_W[index]->Fill(residualRPhi);
		hResidualPhiDT_W[index]->Fill(residualPhi);
		hResidualZDT_W[index]->Fill(residualZ);
		index=wheel*4+station+7;
                hResidualRPhiDT_MB[index]->Fill(residualRPhi);
                hResidualPhiDT_MB[index]->Fill(residualPhi);
                hResidualZDT_MB[index]->Fill(residualZ);

	    }
	    else if(det==2){ //CSC
	    sprintf(nameOfHistoRPhi, "ResidualRPhi_ME%ldR%1dCh%1d",station,ring,chamber );
	    sprintf(nameOfHistoPhi, "ResidualPhi_ME%ldR%1dCh%1d",station,ring,chamber);
	    sprintf(nameOfHistoZ, "ResidualZ_ME%ldR%1dCh%1d",station,ring,chamber);
	    hResidualRPhiCSC->Fill(residualRPhi);
	    hResidualPhiCSC->Fill(residualPhi);
	    hResidualZCSC->Fill(residualZ);

		int index=2*station+ring+7;
			if(ring==4) ring=1;
		if(station==-1) index=5+ring;
		if(station==1) index=8+ring;
                hResidualRPhiCSC_ME[index]->Fill(residualRPhi);
                hResidualPhiCSC_ME[index]->Fill(residualPhi);
		hResidualZCSC_ME[index]->Fill(residualZ);	    		

		}		    
	    
	    TH1F *histoRPhi = new TH1F(nameOfHistoRPhi, nameOfHistoRPhi, 100, -2.0*range, 2.0*range);
	    TH1F *histoPhi = new TH1F(nameOfHistoPhi, nameOfHistoPhi, 100, -0.1*range, 0.1*range);
	    TH1F *histoZ = new TH1F(nameOfHistoZ, nameOfHistoZ, 100, -2.0*range, 2.0*range);

	    histoRPhi->Fill(residualRPhi);
	    histoPhi->Fill(residualPhi);
	    histoZ->Fill(residualZ);	
	    //Push them into their respective vectors
	    unitsRPhi.push_back(histoRPhi);
	    unitsPhi.push_back(histoPhi);
	    unitsZ.push_back(histoZ);

	    } 
	    else {
	    //If the detector was not new, just fill the histogram
	    unitsRPhi.at(position)->Fill(residualRPhi);
	    unitsPhi.at(position)->Fill(residualPhi);
	    unitsZ.at(position)->Fill(residualZ);
		
            if(det==1){ // DT
            hResidualRPhiDT->Fill(residualRPhi);
            hResidualPhiDT->Fill(residualPhi);
            hResidualZDT->Fill(residualZ);
	    	int index = wheel+2;
		hResidualRPhiDT_W[index]->Fill(residualRPhi);
		hResidualPhiDT_W[index]->Fill(residualPhi);
		hResidualZDT_W[index]->Fill(residualZ);
		index=wheel*4+station+7;
            	hResidualRPhiDT_MB[index]->Fill(residualRPhi);
                hResidualPhiDT_MB[index]->Fill(residualPhi);
                hResidualZDT_MB[index]->Fill(residualZ);
	    }
	    else if(det==2){ //CSC
            hResidualRPhiCSC->Fill(residualRPhi);
            hResidualPhiCSC->Fill(residualPhi);
            hResidualZCSC->Fill(residualZ);
	    	int index=2*station+ring+7;
		if(ring==4) ring=1;
		if(station==-1) index=5+ring;
		if(station==1) index=8+ring;
                hResidualRPhiCSC_ME[index]->Fill(residualRPhi);
                hResidualPhiCSC_ME[index]->Fill(residualPhi);
		hResidualZCSC_ME[index]->Fill(residualZ);	    		
	    }	
	    
	  }
	  countPoints++;
	
 	  innerTSOS = destiny;


	}catch(...) {
	  edm::LogError("MuonAlignmentAnalyzer")<<" Error!! Exception in propagator catched" << endl;
	  continue;
	}

	} //loop over my4DTrack
	} //TSOS was valid

	} // loop over recHitsSize

	numberOfHits=numberOfHits+countPoints;
	} //loop over STAtracks


 } //end doResplots
 

}


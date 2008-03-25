/** \class MuonAlignmentAnalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2007/07/19 17:53:06 $
 *  $Revision: 1.5 $
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
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

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
  
  theDataType = pset.getUntrackedParameter<string>("DataType");
  ptRangeMin = pset.getUntrackedParameter<unsigned int>("ptRangeMin");
  ptRangeMax = pset.getUntrackedParameter<unsigned int>("ptRangeMax");
  invMassRangeMin = pset.getUntrackedParameter<unsigned int>("invMassRangeMin");
  invMassRangeMax = pset.getUntrackedParameter<unsigned int>("invMassRangeMax");

  doSAplots = pset.getUntrackedParameter<bool>("doSAplots");
  doGBplots = pset.getUntrackedParameter<bool>("doGBplots");
  doResplots = pset.getUntrackedParameter<bool>("doResplots");
 
  if(theDataType != "RealData" && theDataType != "SimData")
    edm::LogError("MuonAlignmentAnalyzer")  << "Error in Data Type!!"<<endl;

  numberOfSimTracks=0;
  numberOfSARecTracks=0;
  numberOfGBRecTracks=0;
  numberOfHits=0;
}

/// Destructor
MuonAlignmentAnalyzer::~MuonAlignmentAnalyzer(){
}

void MuonAlignmentAnalyzer::beginJob(const EventSetup& eventSetup){

  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);
  
  //Create the propagator
  if(doResplots)  thePropagator = new SteppingHelixPropagator(&*theMGField, alongMomentum);


// Define and book histograms for SA and GB muon quantities/objects

  if(doGBplots) {
  hGBNmuons = fs->make<TH1F>("GBNmuons","Nmuons",10,0,10);
  hGBNmuons_Barrel = fs->make<TH1F>("GBNmuons_Barrel","Nmuons",10,0,10);
  hGBNmuons_Endcap = fs->make<TH1F>("GBNmuons_Endcap","Nmuons",10,0,10);
  hGBNhits = fs->make<TH1F>("GBNhits","Nhits",100,0,100);
  hGBNhits_Barrel = fs->make<TH1F>("GBNhits_Barrel","Nhits",100,0,100);
  hGBNhits_Endcap = fs->make<TH1F>("GBNhits_Endcap","Nhits",100,0,100);
  hGBPTRec = fs->make<TH1F>("GBpTRec","p_{T}^{rec}",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hGBPTRec_Barrel = fs->make<TH1F>("GBpTRec_Barrel","p_{T}^{rec}",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hGBPTRec_Endcap = fs->make<TH1F>("GBpTRec_Endcap","p_{T}^{rec}",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hGBPTvsEta = fs->make<TH2F> ("GBPTvsEta","p_{T}^{rec} VS #eta",100,-2.5,2.5,ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hGBPTvsPhi = fs->make<TH2F> ("GBPTvsPhi","p_{T}^{rec} VS #phi",100,-3.1416,3.1416,ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hGBPhivsEta = fs->make<TH2F> ("GBPhivsEta","#phi VS #eta",100,-2.5,2.5,100,-3.1416,3.1416);

  if(theDataType == "SimData"){
  hGBPTDiff = fs->make<TH1F>("GBpTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);
  hGBPTDiffvsEta = fs->make<TH2F> ("GBPTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hGBPTDiffvsPhi = fs->make<TH2F> ("GBPTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-3.1416,3.1416,250,-120,120);
  hGBPTres = fs->make<TH1F>("GBpTRes","pT Resolution",100,-2,2);
  hGBPTres_Barrel = fs->make<TH1F>("GBpTRes_Barrel","pT Resolution",100,-2,2);
  hGBPTres_Endcap = fs->make<TH1F>("GBpTRes_Endcap","pT Resolution",100,-2,2);
  hGBinvPTres = fs->make<TH1F>("GBinvPTRes","#sigma (q/p_{T}) Resolution",100,-2,2);  
  hGBinvPTvsEta = fs->make<TH2F> ("GBinvPTvsEta","#sigma (q/p_{T}) VS #eta",100,-2.5,2.5,100,-2,2);
  hGBinvPTvsPhi = fs->make<TH2F> ("GBinvPTvsPhi","#sigma (q/p_{T}) VS #phi",100,-3.1416,3.1416,100,-2,2);
  hGBinvPTvsNhits = fs->make<TH2F> ("GBinvPTvsNhits","#sigma (q/p_{T}) VS Nhits",100,0,100,100,-2,2);
  }
  
  hGBChi2 = fs->make<TH1F>("GBChi2","Chi2",200,0,200);
  hGBChi2_Barrel = fs->make<TH1F>("GBChi2_Barrel","Chi2",200,0,200);
  hGBChi2_Endcap  = fs->make<TH1F>("GBChi2_Endcap ","Chi2",200,0,200);
  hGBInvM = fs->make<TH1F>("GBInvM","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hGBInvM_Barrel = fs->make<TH1F>("GBInvM_Barrel","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hGBInvM_Endcap  = fs->make<TH1F>("GBInvM_Endcap ","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hGBInvM_Overlap = fs->make<TH1F>("GBInvM_Overlap","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  }
  
  
  if(doSAplots) {
  hSANmuons = fs->make<TH1F>("SANmuons","Nmuons",10,0,10);
  hSANmuons_Barrel = fs->make<TH1F>("SANmuons_Barrel","Nmuons",10,0,10);
  hSANmuons_Endcap = fs->make<TH1F>("SANmuons_Endcap","Nmuons",10,0,10);
  hSANhits = fs->make<TH1F>("SANhits","Nhits",100,0,100);
  hSANhits_Barrel = fs->make<TH1F>("SANhits_Barrel","Nhits",100,0,100);
  hSANhits_Endcap = fs->make<TH1F>("SANhits_Endcap","Nhits",100,0,100);
  hSAPTRec = fs->make<TH1F>("SApTRec","p_{T}^{rec}",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSAPTRec_Barrel = fs->make<TH1F>("SApTRec_Barrel","p_{T}^{rec}",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSAPTRec_Endcap = fs->make<TH1F>("SApTRec_Endcap","p_{T}^{rec}",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSAPTvsEta = fs->make<TH2F> ("SAPTvsEta","p_{T}^{rec} VS #eta",100,-2.5,2.5,ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSAPTvsPhi = fs->make<TH2F> ("SAPTvsPhi","p_{T}^{rec} VS #phi",100,-3.1416,3.1416,ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSAPhivsEta = fs->make<TH2F> ("SAPhivsEta","#phi VS #eta",100,-2.5,2.5,100,-3.1416,3.1416);

  if(theDataType == "SimData"){
  hSAPTDiff = fs->make<TH1F>("SApTDiff","p_{T}^{rec} - p_{T}^{gen} ",250,-120,120);
  hSAPTDiffvsEta = fs->make<TH2F> ("SAPTDiffvsEta","p_{T}^{rec} - p_{T}^{gen} VS #eta",100,-2.5,2.5,250,-120,120);
  hSAPTDiffvsPhi = fs->make<TH2F> ("SAPTDiffvsPhi","p_{T}^{rec} - p_{T}^{gen} VS #phi",100,-3.1416,3.1416,250,-120,120);
  hSAPTres = fs->make<TH1F>("SApTRes","pT Resolution",100,-2,2);
  hSAPTres_Barrel = fs->make<TH1F>("SApTRes_Barrel","pT Resolution",100,-2,2);
  hSAPTres_Endcap = fs->make<TH1F>("SApTRes_Endcap","pT Resolution",100,-2,2);
  hSAinvPTres = fs->make<TH1F>("SAinvPTRes","1/pT Resolution",100,-2,2);

  hSAinvPTvsEta = fs->make<TH2F> ("SAinvPTvsEta","#sigma (q/p_{T}) VS #eta",100,-2.5,2.5,100,-2,2);
  hSAinvPTvsPhi = fs->make<TH2F> ("SAinvPTvsPhi","#sigma (q/p_{T}) VS #phi",100,-3.1416,3.1416,100,-2,2);
  hSAinvPTvsNhits = fs->make<TH2F> ("SAinvPTvsNhits","#sigma (q/p_{T}) VS Nhits",100,0,100,100,-2,2);
}
  hSAInvM = fs->make<TH1F>("SAInvM","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSAInvM_Barrel = fs->make<TH1F>("SAInvM_Barrel","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSAInvM_Endcap = fs->make<TH1F>("SAInvM_Endcap","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSAInvM_Overlap = fs->make<TH1F>("SAInvM_Overlap","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSAChi2 = fs->make<TH1F>("SAChi2","Chi2",200,0,200);
  hSAChi2_Barrel = fs->make<TH1F>("SAChi2_Barrel","Chi2",200,0,200);
  hSAChi2_Endcap = fs->make<TH1F>("SAChi2_Endcap","Chi2",200,0,200);
  }


  if(theDataType == "SimData"){
  hSimNmuons = fs->make<TH1F>("SimNmuons","Nmuons",10,0,10);
  hSimNmuons_Barrel = fs->make<TH1F>("SimNmuons_Barrel","Nmuons",10,0,10);
  hSimNmuons_Endcap = fs->make<TH1F>("SimNmuons_Endcap","Nmuons",10,0,10);
  hSimPT = fs->make<TH1F>("SimPT","p_{T}^{gen} ",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSimPT_Barrel = fs->make<TH1F>("SimPT_Barrel","p_{T}^{gen} ",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSimPT_Endcap = fs->make<TH1F>("SimPT_Endcap","p_{T}^{gen} ",ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSimPTvsEta = fs->make<TH2F> ("SimPTvsEta","p_{T}^{gen} VS #eta",100,-2.5,2.5,ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSimPTvsPhi = fs->make<TH2F> ("SimPTvsPhi","p_{T}^{gen} VS #phi",100,-3.1416,3.1416,ptRangeMax-ptRangeMin,ptRangeMin,ptRangeMax);
  hSimPhivsEta = fs->make<TH2F> ("SimPhivsEta","#phi VS #eta",100,-2.5,2.5,100,-3.1416,3.1416);
  hSimInvM = fs->make<TH1F>("SimInvM","M_{inv}^{gen} ",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSimInvM_Barrel = fs->make<TH1F>("SimInvM_Barrel","M_{inv}^{rec}",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSimInvM_Endcap = fs->make<TH1F>("SimInvM_Endcap","M_{inv}^{gen} ",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  hSimInvM_Overlap = fs->make<TH1F>("SimInvM_Overlap","M_{inv}^{gen} ",invMassRangeMax-invMassRangeMin,invMassRangeMin,invMassRangeMax);
  }

  if(doResplots){
// All DT and CSC chambers 
  hResidualRPhiDT = fs->make<TH1F>("hResidualRPhiDT","hResidualRPhiDT",200,-10,10);
  hResidualPhiDT = fs->make<TH1F>("hResidualPhiDT","hResidualPhiDT",100,-1,1);
  hResidualZDT = fs->make<TH1F>("hResidualZDT","hResidualZDT",200,-10,10);
  hResidualRPhiCSC = fs->make<TH1F>("hResidualRPhiCSC","hResidualRPhiCSC",200,-10,10);
  hResidualPhiCSC = fs->make<TH1F>("hResidualPhiCSC","hResidualPhiCSC",100,-1,1);
  hResidualZCSC = fs->make<TH1F>("hResidualZCSC","hResidualZCSC",200,-10,10);

// DT Wheels
  hResidualRPhiDT_W[0]=fs->make<TH1F>("hResidualRPhiDT_W-2","hResidualRPhiDT_W-2",200,-10,10);
  hResidualPhiDT_W[0]=fs->make<TH1F>("hResidualPhiDT_W-2","hResidualPhiDT_W-2",200,-1,1);
  hResidualZDT_W[0] = fs->make<TH1F>("hResidualZDT_W-2","hResidualZDT_W-2",200,-10,10);
  hResidualRPhiDT_W[1]=fs->make<TH1F>("hResidualRPhiDT_W-1","hResidualRPhiDT_W-1",200,-10,10);
  hResidualPhiDT_W[1]=fs->make<TH1F>("hResidualPhiDT_W-1","hResidualPhiDT_W-1",200,-1,1);
  hResidualZDT_W[1] = fs->make<TH1F>("hResidualZDT_W-1","hResidualZDT_W-1",200,-10,10);
  hResidualRPhiDT_W[2]=fs->make<TH1F>("hResidualRPhiDT_W0","hResidualRPhiDT_W0",200,-10,10);
  hResidualPhiDT_W[2]=fs->make<TH1F>("hResidualPhiDT_W0","hResidualPhiDT_W0",200,-1,1);
  hResidualZDT_W[2] = fs->make<TH1F>("hResidualZDT_W0","hResidualZDT_W0",200,-10,10);
  hResidualRPhiDT_W[3]=fs->make<TH1F>("hResidualRPhiDT_W1","hResidualRPhiDT_W1",200,-10,10);
  hResidualPhiDT_W[3]=fs->make<TH1F>("hResidualPhiDT_W1","hResidualPhiDT_W1",200,-1,1);
  hResidualZDT_W[3] = fs->make<TH1F>("hResidualZDT_W1","hResidualZDT_W1",200,-10,10);
  hResidualRPhiDT_W[4]=fs->make<TH1F>("hResidualRPhiDT_W2","hResidualRPhiDT_W2",200,-10,10);
  hResidualPhiDT_W[4]=fs->make<TH1F>("hResidualPhiDT_W2","hResidualPhiDT_W2",200,-1,1);
  hResidualZDT_W[4] = fs->make<TH1F>("hResidualZDT_W2","hResidualZDT_W2",200,-10,10);

// DT Stations
  hResidualRPhiDT_MB[0]=fs->make<TH1F>("hResidualRPhiDT_MB-2/1","hResidualRPhiDT_MB-2/1",200,-10,10);
  hResidualPhiDT_MB[0]=fs->make<TH1F>("hResidualPhiDT_MB-2/1","hResidualPhiDT_MB-2/1",200,-1,1);
  hResidualZDT_MB[0] = fs->make<TH1F>("hResidualZDT_MB-2/1","hResidualZDT_MB-2/1",200,-10,10);
  hResidualRPhiDT_MB[1]=fs->make<TH1F>("hResidualRPhiDT_MB-2/2","hResidualRPhiDT_MB-2/2",200,-10,10);
  hResidualPhiDT_MB[1]=fs->make<TH1F>("hResidualPhiDT_MB-2/2","hResidualPhiDT_MB-2/2",200,-1,1);
  hResidualZDT_MB[1] = fs->make<TH1F>("hResidualZDT_MB-2/2","hResidualZDT_MB-2/2",200,-10,10);
  hResidualRPhiDT_MB[2]=fs->make<TH1F>("hResidualRPhiDT_MB-2/3","hResidualRPhiDT_MB-2/3",200,-10,10);
  hResidualPhiDT_MB[2]=fs->make<TH1F>("hResidualPhiDT_MB-2/3","hResidualPhiDT_MB-2/3",200,-1,1);
  hResidualZDT_MB[2] = fs->make<TH1F>("hResidualZDT_MB-2/3","hResidualZDT_MB-2/3",200,-10,10);
  hResidualRPhiDT_MB[3]=fs->make<TH1F>("hResidualRPhiDT_MB-2/4","hResidualRPhiDT_MB-2/4",200,-10,10);
  hResidualPhiDT_MB[3]=fs->make<TH1F>("hResidualPhiDT_MB-2/4","hResidualPhiDT_MB-2/4",200,-1,1);
  hResidualZDT_MB[3] = fs->make<TH1F>("hResidualZDT_MB-2/4","hResidualZDT_MB-2/4",200,-10,10);
  hResidualRPhiDT_MB[4]=fs->make<TH1F>("hResidualRPhiDT_MB-1/1","hResidualRPhiDT_MB-1/1",200,-10,10);
  hResidualPhiDT_MB[4]=fs->make<TH1F>("hResidualPhiDT_MB-1/1","hResidualPhiDT_MB-1/1",200,-1,1);
  hResidualZDT_MB[4] = fs->make<TH1F>("hResidualZDT_MB-1/1","hResidualZDT_MB-1/1",200,-10,10);
  hResidualRPhiDT_MB[5]=fs->make<TH1F>("hResidualRPhiDT_MB-1/2","hResidualRPhiDT_MB-1/2",200,-10,10);
  hResidualPhiDT_MB[5]=fs->make<TH1F>("hResidualPhiDT_MB-1/2","hResidualPhiDT_MB-1/2",200,-1,1);
  hResidualZDT_MB[5] = fs->make<TH1F>("hResidualZDT_MB-1/2","hResidualZDT_MB-1/2",200,-10,10);
  hResidualRPhiDT_MB[6]=fs->make<TH1F>("hResidualRPhiDT_MB-1/3","hResidualRPhiDT_MB-1/3",200,-10,10);
  hResidualPhiDT_MB[6]=fs->make<TH1F>("hResidualPhiDT_MB-1/3","hResidualPhiDT_MB-1/3",200,-1,1);
  hResidualZDT_MB[6] = fs->make<TH1F>("hResidualZDT_MB-1/3","hResidualZDT_MB-1/3",200,-10,10);
  hResidualRPhiDT_MB[7]=fs->make<TH1F>("hResidualRPhiDT_MB-1/4","hResidualRPhiDT_MB-1/4",200,-10,10);
  hResidualPhiDT_MB[7]=fs->make<TH1F>("hResidualPhiDT_MB-1/4","hResidualPhiDT_MB-1/4",200,-1,1);
  hResidualZDT_MB[7] = fs->make<TH1F>("hResidualZDT_MB-1/4","hResidualZDT_MB-1/4",200,-10,10);
  hResidualRPhiDT_MB[8]=fs->make<TH1F>("hResidualRPhiDT_MB0/1","hResidualRPhiDT_MB0/1",200,-10,10);
  hResidualPhiDT_MB[8]=fs->make<TH1F>("hResidualPhiDT_MB0/1","hResidualPhiDT_MB0/1",200,-1,1);
  hResidualZDT_MB[8] = fs->make<TH1F>("hResidualZDT_MB0/1","hResidualZDT_MB0/1",200,-10,10);
  hResidualRPhiDT_MB[9]=fs->make<TH1F>("hResidualRPhiDT_MB0/2","hResidualRPhiDT_MB0/2",200,-10,10);
  hResidualPhiDT_MB[9]=fs->make<TH1F>("hResidualPhiDT_MB0/2","hResidualPhiDT_MB0/2",200,-1,1);
  hResidualZDT_MB[9] = fs->make<TH1F>("hResidualZDT_MB0/2","hResidualZDT_MB0/2",200,-10,10);
  hResidualRPhiDT_MB[10]=fs->make<TH1F>("hResidualRPhiDT_MB0/3","hResidualRPhiDT_MB0/3",200,-10,10);
  hResidualPhiDT_MB[10]=fs->make<TH1F>("hResidualPhiDT_MB0/3","hResidualPhiDT_MB0/3",200,-1,1);
  hResidualZDT_MB[10] = fs->make<TH1F>("hResidualZDT_MB0/3","hResidualZDT_MB0/3",200,-10,10);
  hResidualRPhiDT_MB[11]=fs->make<TH1F>("hResidualRPhiDT_MB0/4","hResidualRPhiDT_MB0/4",200,-10,10);
  hResidualPhiDT_MB[11]=fs->make<TH1F>("hResidualPhiDT_MB0/4","hResidualPhiDT_MB0/4",200,-1,1);
  hResidualZDT_MB[11] = fs->make<TH1F>("hResidualZDT_MB0/4","hResidualZDT_MB0/4",200,-10,10);
  hResidualRPhiDT_MB[12]=fs->make<TH1F>("hResidualRPhiDT_MB1/1","hResidualRPhiDT_MB1/1",200,-10,10);
  hResidualPhiDT_MB[12]=fs->make<TH1F>("hResidualPhiDT_MB1/1","hResidualPhiDT_MB1/1",200,-1,1);
  hResidualZDT_MB[12] = fs->make<TH1F>("hResidualZDT_MB1/1","hResidualZDT_MB1/1",200,-10,10);
  hResidualRPhiDT_MB[13]=fs->make<TH1F>("hResidualRPhiDT_MB1/2","hResidualRPhiDT_MB1/2",200,-10,10);
  hResidualPhiDT_MB[13]=fs->make<TH1F>("hResidualPhiDT_MB1/2","hResidualPhiDT_MB1/2",200,-1,1);
  hResidualZDT_MB[13] = fs->make<TH1F>("hResidualZDT_MB1/2","hResidualZDT_MB1/2",200,-10,10);
  hResidualRPhiDT_MB[14]=fs->make<TH1F>("hResidualRPhiDT_MB1/3","hResidualRPhiDT_MB1/3",200,-10,10);
  hResidualPhiDT_MB[14]=fs->make<TH1F>("hResidualPhiDT_MB1/3","hResidualPhiDT_MB1/3",200,-1,1);
  hResidualZDT_MB[14] = fs->make<TH1F>("hResidualZDT_MB1/3","hResidualZDT_MB1/3",200,-10,10);
  hResidualRPhiDT_MB[15]=fs->make<TH1F>("hResidualRPhiDT_MB1/4","hResidualRPhiDT_MB1/4",200,-10,10);
  hResidualPhiDT_MB[15]=fs->make<TH1F>("hResidualPhiDT_MB1/4","hResidualPhiDT_MB1/4",200,-1,1);
  hResidualZDT_MB[15] = fs->make<TH1F>("hResidualZDT_MB1/4","hResidualZDT_MB1/4",200,-10,10);
  hResidualRPhiDT_MB[16]=fs->make<TH1F>("hResidualRPhiDT_MB2/1","hResidualRPhiDT_MB2/1",200,-10,10);
  hResidualPhiDT_MB[16]=fs->make<TH1F>("hResidualPhiDT_MB2/1","hResidualPhiDT_MB2/1",200,-1,1);
  hResidualZDT_MB[16] = fs->make<TH1F>("hResidualZDT_MB2/1","hResidualZDT_MB2/1",200,-10,10);
  hResidualRPhiDT_MB[17]=fs->make<TH1F>("hResidualRPhiDT_MB2/2","hResidualRPhiDT_MB2/2",200,-10,10);
  hResidualPhiDT_MB[17]=fs->make<TH1F>("hResidualPhiDT_MB2/2","hResidualPhiDT_MB2/2",200,-1,1);
  hResidualZDT_MB[17] = fs->make<TH1F>("hResidualZDT_MB2/2","hResidualZDT_MB2/2",200,-10,10);
  hResidualRPhiDT_MB[18]=fs->make<TH1F>("hResidualRPhiDT_MB2/3","hResidualRPhiDT_MB2/3",200,-10,10);
  hResidualPhiDT_MB[18]=fs->make<TH1F>("hResidualPhiDT_MB2/3","hResidualPhiDT_MB2/3",200,-1,1);
  hResidualZDT_MB[18] = fs->make<TH1F>("hResidualZDT_MB2/3","hResidualZDT_MB2/3",200,-10,10);
  hResidualRPhiDT_MB[19]=fs->make<TH1F>("hResidualRPhiDT_MB2/4","hResidualRPhiDT_MB2/4",200,-10,10);
  hResidualPhiDT_MB[19]=fs->make<TH1F>("hResidualPhiDT_MB2/4","hResidualPhiDT_MB2/4",200,-1,1);
  hResidualZDT_MB[19] = fs->make<TH1F>("hResidualZDT_MB2/4","hResidualZDT_MB2/4",200,-10,10);


// CSC Stations
  hResidualRPhiCSC_ME[0]=fs->make<TH1F>("hResidualRPhiCSC_ME-4/1","hResidualRPhiCSC_ME-4/1",200,-10,10);
  hResidualPhiCSC_ME[0]=fs->make<TH1F>("hResidualPhiCSC_ME-4/1","hResidualPhiCSC_ME-4/1",200,-1,1);
  hResidualZCSC_ME[0] = fs->make<TH1F>("hResidualZCSC_ME-4/1","hResidualZCSC_ME-4/1",200,-10,10);
  hResidualRPhiCSC_ME[1]=fs->make<TH1F>("hResidualRPhiCSC_ME-4/2","hResidualRPhiCSC_ME-4/2",200,-10,10);
  hResidualPhiCSC_ME[1]=fs->make<TH1F>("hResidualPhiCSC_ME-4/2","hResidualPhiCSC_ME-4/2",200,-1,1);
  hResidualZCSC_ME[1] = fs->make<TH1F>("hResidualZCSC_ME-4/2","hResidualZCSC_ME-4/2",200,-10,10);
  hResidualRPhiCSC_ME[2]=fs->make<TH1F>("hResidualRPhiCSC_ME-3/1","hResidualRPhiCSC_ME-3/1",200,-10,10);
  hResidualPhiCSC_ME[2]=fs->make<TH1F>("hResidualPhiCSC_ME-3/1","hResidualPhiCSC_ME-3/1",200,-1,1);
  hResidualZCSC_ME[2] = fs->make<TH1F>("hResidualZCSC_ME-3/1","hResidualZCSC_ME-3/1",200,-10,10);
  hResidualRPhiCSC_ME[3]=fs->make<TH1F>("hResidualRPhiCSC_ME-3/2","hResidualRPhiCSC_ME-3/2",200,-10,10);
  hResidualPhiCSC_ME[3]=fs->make<TH1F>("hResidualPhiCSC_ME-3/2","hResidualPhiCSC_ME-3/2",200,-1,1);
  hResidualZCSC_ME[3] = fs->make<TH1F>("hResidualZCSC_ME-3/2","hResidualZCSC_ME-3/2",200,-10,10);
  hResidualRPhiCSC_ME[4]=fs->make<TH1F>("hResidualRPhiCSC_ME-2/1","hResidualRPhiCSC_ME-2/1",200,-10,10);
  hResidualPhiCSC_ME[4]=fs->make<TH1F>("hResidualPhiCSC_ME-2/1","hResidualPhiCSC_ME-2/1",200,-1,1);
  hResidualZCSC_ME[4] = fs->make<TH1F>("hResidualZCSC_ME-2/1","hResidualZCSC_ME-2/1",200,-10,10);
  hResidualRPhiCSC_ME[5]=fs->make<TH1F>("hResidualRPhiCSC_ME-2/2","hResidualRPhiCSC_ME-2/2",200,-10,10);
  hResidualPhiCSC_ME[5]=fs->make<TH1F>("hResidualPhiCSC_ME-2/2","hResidualPhiCSC_ME-2/2",200,-1,1);
  hResidualZCSC_ME[5] = fs->make<TH1F>("hResidualZCSC_ME-2/2","hResidualZCSC_ME-2/2",200,-10,10);
  hResidualRPhiCSC_ME[6]=fs->make<TH1F>("hResidualRPhiCSC_ME-1/1","hResidualRPhiCSC_ME-1/1",200,-10,10);
  hResidualPhiCSC_ME[6]=fs->make<TH1F>("hResidualPhiCSC_ME-1/1","hResidualPhiCSC_ME-1/1",200,-1,1);
  hResidualZCSC_ME[6] = fs->make<TH1F>("hResidualZCSC_ME-1/1","hResidualZCSC_ME-1/1",200,-10,10);
  hResidualRPhiCSC_ME[7]=fs->make<TH1F>("hResidualRPhiCSC_ME-1/2","hResidualRPhiCSC_ME-1/2",200,-10,10);
  hResidualPhiCSC_ME[7]=fs->make<TH1F>("hResidualPhiCSC_ME-1/2","hResidualPhiCSC_ME-1/2",200,-1,1);
  hResidualZCSC_ME[7] = fs->make<TH1F>("hResidualZCSC_ME-1/2","hResidualZCSC_ME-1/2",200,-10,10);
  hResidualRPhiCSC_ME[8]=fs->make<TH1F>("hResidualRPhiCSC_ME-1/3","hResidualRPhiCSC_ME-1/3",200,-10,10);
  hResidualPhiCSC_ME[8]=fs->make<TH1F>("hResidualPhiCSC_ME-1/3","hResidualPhiCSC_ME-1/3",200,-1,1);
  hResidualZCSC_ME[8] = fs->make<TH1F>("hResidualZCSC_ME-1/3","hResidualZCSC_ME-1/3",200,-10,10);
  hResidualRPhiCSC_ME[9]=fs->make<TH1F>("hResidualRPhiCSC_ME1/1","hResidualRPhiCSC_ME1/1",200,-10,10);
  hResidualPhiCSC_ME[9]=fs->make<TH1F>("hResidualPhiCSC_ME1/1","hResidualPhiCSC_ME1/1",100,-1,1);
  hResidualZCSC_ME[9] = fs->make<TH1F>("hResidualZCSC_ME1/1","hResidualZCSC_ME1/1",200,-10,10);
  hResidualRPhiCSC_ME[10]=fs->make<TH1F>("hResidualRPhiCSC_ME1/2","hResidualRPhiCSC_ME1/2",200,-10,10);
  hResidualPhiCSC_ME[10]=fs->make<TH1F>("hResidualPhiCSC_ME1/2","hResidualPhiCSC_ME1/2",200,-1,1);
  hResidualZCSC_ME[10] = fs->make<TH1F>("hResidualZCSC_ME1/2","hResidualZCSC_ME1/2",200,-10,10);
  hResidualRPhiCSC_ME[11]=fs->make<TH1F>("hResidualRPhiCSC_ME1/3","hResidualRPhiCSC_ME1/3",200,-10,10);
  hResidualPhiCSC_ME[11]=fs->make<TH1F>("hResidualPhiCSC_ME1/3","hResidualPhiCSC_ME1/3",200,-1,1);
  hResidualZCSC_ME[11] = fs->make<TH1F>("hResidualZCSC_ME1/3","hResidualZCSC_ME1/3",200,-10,10);
  hResidualRPhiCSC_ME[12]=fs->make<TH1F>("hResidualRPhiCSC_ME2/1","hResidualRPhiCSC_ME2/1",200,-10,10);
  hResidualPhiCSC_ME[12]=fs->make<TH1F>("hResidualPhiCSC_ME2/1","hResidualPhiCSC_ME2/1",200,-1,1);
  hResidualZCSC_ME[12] = fs->make<TH1F>("hResidualZCSC_ME2/1","hResidualZCSC_ME2/1",200,-10,10);
  hResidualRPhiCSC_ME[13]=fs->make<TH1F>("hResidualRPhiCSC_ME2/2","hResidualRPhiCSC_ME2/2",200,-10,10);
  hResidualPhiCSC_ME[13]=fs->make<TH1F>("hResidualPhiCSC_ME2/2","hResidualPhiCSC_ME2/2",200,-1,1);
  hResidualZCSC_ME[13] = fs->make<TH1F>("hResidualZCSC_ME2/2","hResidualZCSC_ME2/2",200,-10,10);
  hResidualRPhiCSC_ME[14]=fs->make<TH1F>("hResidualRPhiCSC_ME3/1","hResidualRPhiCSC_ME3/1",200,-10,10);
  hResidualPhiCSC_ME[14]=fs->make<TH1F>("hResidualPhiCSC_ME3/1","hResidualPhiCSC_ME3/1",200,-1,1);
  hResidualZCSC_ME[14] = fs->make<TH1F>("hResidualZCSC_ME3/1","hResidualZCSC_ME3/1",200,-10,10);
  hResidualRPhiCSC_ME[15]=fs->make<TH1F>("hResidualRPhiCSC_ME3/2","hResidualRPhiCSC_ME3/2",200,-10,10);
  hResidualPhiCSC_ME[15]=fs->make<TH1F>("hResidualPhiCSC_ME3/2","hResidualPhiCSC_ME3/2",200,-1,1);
  hResidualZCSC_ME[15] = fs->make<TH1F>("hResidualZCSC_ME3/2","hResidualZCSC_ME3/2",200,-10,10);
  hResidualRPhiCSC_ME[16]=fs->make<TH1F>("hResidualRPhiCSC_ME4/1","hResidualRPhiCSC_ME4/1",200,-10,10);
  hResidualPhiCSC_ME[16]=fs->make<TH1F>("hResidualPhiCSC_ME4/1","hResidualPhiCSC_ME4/1",200,-1,1);
  hResidualZCSC_ME[16] = fs->make<TH1F>("hResidualZCSC_ME4/1","hResidualZCSC_ME4/1",200,-10,10);
  hResidualRPhiCSC_ME[17]=fs->make<TH1F>("hResidualRPhiCSC_ME4/2","hResidualRPhiCSC_ME4/2",200,-10,10);
  hResidualPhiCSC_ME[17]=fs->make<TH1F>("hResidualPhiCSC_ME4/2","hResidualPhiCSC_ME4/2",200,-1,1);
  hResidualZCSC_ME[17] = fs->make<TH1F>("hResidualZCSC_ME4/2","hResidualZCSC_ME4/2",200,-10,10);
  }

}

void MuonAlignmentAnalyzer::endJob(){


    edm::LogInfo("MuonAlignmentAnalyzer")  << "----------------- " << endl << endl;

  if(theDataType == "SimData")
    edm::LogInfo("MuonAlignmentAnalyzer")  << "Number of Sim tracks: " << numberOfSimTracks << endl << endl;

  if(doSAplots)
    edm::LogInfo("MuonAlignmentAnalyzer")  << "Number of SA Reco tracks: " << numberOfSARecTracks << endl << endl;

  if(doGBplots)
  edm::LogInfo("MuonAlignmentAnalyzer")  << "Number of GB Reco tracks: " << numberOfGBRecTracks << endl << endl;

  if(doResplots){

  delete thePropagator;

  edm::LogInfo("MuonAlignmentAnalyzer")  << "Number of Hits considered for residuals: " << numberOfHits << endl << endl;

  }

}
 

void MuonAlignmentAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
  

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  
  GlobalVector p1,p2;
  std::vector< double > simPar[4] ; //pt,eta,phi,charge
  

// ######### if data= MC, do Simulation Plots#####
  if(theDataType == "SimData"){
  double simEta=0;
  double simPt=0;
  double simPhi=0;
  int i=0, ie=0,ib=0;

  // Get the SimTrack collection from the event
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
  

    SimTrackContainer::const_iterator simTrack;

    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
      if (abs((*simTrack).type()) == 13) {
	i++;
	simPt=(*simTrack).momentum().perp();
	simEta=(*simTrack).momentum().eta();
	simPhi=(*simTrack).momentum().phi();
	numberOfSimTracks++;
	hSimPT->Fill(simPt);
	if(abs(simEta)<1.04) {hSimPT_Barrel->Fill(simPt);ib++;}
	else {hSimPT_Endcap->Fill(simPt);ie++;}
	hSimPTvsEta->Fill(simEta,simPt);
	hSimPTvsPhi->Fill(simPhi,simPt);
	hSimPhivsEta->Fill(simEta,simPhi);

        simPar[0].push_back(simPt);
	simPar[1].push_back(simEta);
	simPar[2].push_back(simPhi);
        simPar[3].push_back((*simTrack).charge());
	
//	Save the muon pair
        if(i==1)  p1=GlobalVector((*simTrack).momentum().x(),(*simTrack).momentum().y(),(*simTrack).momentum().z());
    	if(i==2)  p2=GlobalVector((*simTrack).momentum().x(),(*simTrack).momentum().y(),(*simTrack).momentum().z());
     }    
    }
	hSimNmuons->Fill(i);
	hSimNmuons_Barrel->Fill(ib);
	hSimNmuons_Endcap->Fill(ie);

 if(i>1){ // Take 2 first muons :-(
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hSimInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04) hSimInvM_Barrel->Fill(Minv);
  else if(abs(p1.eta())>=1.04 && abs(p2.eta())>=1.04) hSimInvM_Endcap->Fill(Minv);
  else  hSimInvM_Overlap->Fill(Minv);  
  }

  } //simData
  
  
// ############ Stand Alone Muon plots ###############  
  if(doSAplots){

  double SArecPt=0.;
  double SAeta=0.;
  double SAphi=0.;
  int i=0, ie=0,ib=0;
  double ich=0;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonTag, staTracks);
  numberOfSARecTracks += staTracks->size();

  reco::TrackCollection::const_iterator staTrack;

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    i++;    
    
    SArecPt = (*staTrack).pt();
    SAeta = (*staTrack).eta();
    SAphi = (*staTrack).phi();
    ich= (*staTrack).charge();

    hSAPTRec->Fill(SArecPt);
    hSAPhivsEta->Fill(SAeta,SAphi);
    hSAChi2->Fill((*staTrack).chi2());
    hSANhits->Fill((*staTrack).recHitsSize());
    if(abs(SAeta)<1.04) {hSAPTRec_Barrel->Fill(SArecPt); hSAChi2_Barrel->Fill((*staTrack).chi2()); hSANhits_Barrel->Fill((*staTrack).recHitsSize()); ib++;}
    else {hSAPTRec_Endcap->Fill(SArecPt); hSAChi2_Endcap->Fill((*staTrack).chi2()); hSANhits_Endcap->Fill((*staTrack).recHitsSize()); ie++;}

// save the muon pair
    if(i==1)  p1=GlobalVector((*staTrack).momentum().x(),(*staTrack).momentum().y(),(*staTrack).momentum().z());
    if(i==2)  p2=GlobalVector((*staTrack).momentum().x(),(*staTrack).momentum().y(),(*staTrack).momentum().z());

   
    if(SArecPt && theDataType == "SimData"){  

     double candDeltaR= -999.0, deltaR;
     int iCand=0;
     if(simPar[0].size()>0){
     for(unsigned int  iSim = 0; iSim <simPar[0].size(); iSim++) {
     deltaR=sqrt((SAeta-simPar[1][iSim])*(SAeta-simPar[1][iSim])+(SAphi-simPar[2][iSim])*(SAphi-simPar[2][iSim]));
     if(candDeltaR<0 || deltaR<candDeltaR) {
        candDeltaR=deltaR;
        iCand=iSim;
        }
     }}
	
     double simPt=simPar[0][iCand];
      hSAPTres->Fill( (SArecPt-simPt)/simPt);
    	if(abs(SAeta)<1.04) hSAPTres_Barrel->Fill((SArecPt-simPt)/simPt);
	else hSAPTres_Endcap->Fill((SArecPt-simPt)/simPt);

      hSAPTDiff->Fill(SArecPt-simPt);

      hSAPTDiffvsEta->Fill(SAeta,SArecPt-simPt);
      hSAPTDiffvsPhi->Fill(SAphi,SArecPt-simPt);
	double ptInvRes= ( ich/SArecPt - simPar[3][iCand]/simPt)/ (simPar[3][iCand]/simPt);
      hSAinvPTres->Fill( ptInvRes);

      hSAinvPTvsEta->Fill(SAeta,ptInvRes);
      hSAinvPTvsPhi->Fill(SAphi,ptInvRes);
      hSAinvPTvsNhits->Fill((*staTrack).recHitsSize(),ptInvRes);
      }

      hSAPTvsEta->Fill(SAeta,SArecPt);
      hSAPTvsPhi->Fill(SAphi,SArecPt);
}

	hSANmuons->Fill(i);
	hSANmuons_Barrel->Fill(ib);
	hSANmuons_Endcap->Fill(ie);

  if(i>1){ // Take 2 first muons :-(
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hSAInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04) hSAInvM_Barrel->Fill(Minv);
  else if(abs(p1.eta())>=1.04 && abs(p2.eta())>=1.04) hSAInvM_Endcap->Fill(Minv);
  else hSAInvM_Overlap->Fill(Minv);
  } // 2 first muons

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
  double ich =0;
  int i=0, ie=0,ib=0;

  reco::TrackCollection::const_iterator glbTrack;

  for (glbTrack = glbTracks->begin(); glbTrack != glbTracks->end(); ++glbTrack){
    i++;

    GBrecPt = (*glbTrack).pt(); 
    GBeta = (*glbTrack).eta();
    GBphi = (*glbTrack).phi();
    ich=   (*glbTrack).charge();
    
    hGBPTRec->Fill(GBrecPt);
    hGBPhivsEta->Fill(GBeta,GBphi);
    hGBChi2->Fill((*glbTrack).chi2());
    hGBNhits->Fill((*glbTrack).recHitsSize());
    if(abs(GBeta)<1.04) {hGBPTRec_Barrel->Fill(GBrecPt); hGBChi2_Barrel->Fill((*glbTrack).chi2()); hGBNhits_Barrel->Fill((*glbTrack).recHitsSize()); ib++;}
    else {hGBPTRec_Endcap->Fill(GBrecPt); hGBChi2_Endcap->Fill((*glbTrack).chi2()); hGBNhits_Endcap->Fill((*glbTrack).recHitsSize()); ie++;}
  
// save the muon pair
    if(i==1)  p1=GlobalVector((*glbTrack).momentum().x(),(*glbTrack).momentum().y(),(*glbTrack).momentum().z());
    if(i==2)  p2=GlobalVector((*glbTrack).momentum().x(),(*glbTrack).momentum().y(),(*glbTrack).momentum().z());

    if(GBrecPt && theDataType == "SimData"){ 
     double candDeltaR= -999.0, deltaR;
     int iCand=0;
     if(simPar[0].size()>0){
     for(unsigned int  iSim = 0; iSim <simPar[0].size(); iSim++) {
     deltaR=sqrt((GBeta-simPar[1][iSim])*(GBeta-simPar[1][iSim])+(GBphi-simPar[2][iSim])*(GBphi-simPar[2][iSim]));
     if(candDeltaR<0 || deltaR<candDeltaR) {
	candDeltaR=deltaR;
	iCand=iSim;
	}
     }}

     double simPt=simPar[0][iCand];
	
      hGBPTres->Fill( (GBrecPt-simPt)/simPt);
    	if(abs(GBeta)<1.04) hGBPTres_Barrel->Fill((GBrecPt-simPt)/simPt);
	else hGBPTres_Endcap->Fill((GBrecPt-simPt)/simPt);

      hGBPTDiff->Fill(GBrecPt-simPt);

      hGBPTDiffvsEta->Fill(GBeta,GBrecPt-simPt);
      hGBPTDiffvsPhi->Fill(GBphi,GBrecPt-simPt);

       double ptInvRes= ( ich/GBrecPt - simPar[3][iCand]/simPt)/ (simPar[3][iCand]/simPt);
      hGBinvPTres->Fill( ptInvRes);

      hGBinvPTvsEta->Fill(GBeta,ptInvRes);
      hGBinvPTvsPhi->Fill(GBphi,ptInvRes);
      hGBinvPTvsNhits->Fill((*glbTrack).recHitsSize(),ptInvRes);
    } 


      hGBPTvsEta->Fill(GBeta,GBrecPt);
      hGBPTvsPhi->Fill(GBphi,GBrecPt);

  }

	hGBNmuons->Fill(i);
	hGBNmuons_Barrel->Fill(ib);
	hGBNmuons_Endcap->Fill(ie);

   if(i>1){ // Take 2 first muons :-(
  TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
  TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
  TLorentzVector pair = mu1 + mu2;
  double Minv = pair.M();
  hGBInvM->Fill(Minv);
  if(abs(p1.eta())<1.04 && abs(p2.eta())<1.04)   hGBInvM_Barrel->Fill(Minv);
  else if(abs(p1.eta())>=1.04 && abs(p2.eta())>=1.04) hGBInvM_Endcap->Fill(Minv);
  else hGBInvM_Overlap->Fill(Minv);
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

	// Fill generic histograms
            //If it's a DT
            if(det == 1) {
              DTChamberId myChamber(rawId);
              wheel=myChamber.wheel();
              station = myChamber.station();
              sector=myChamber.sector();

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
	    else if (det==2){
              CSCDetId myChamber(rawId);
              endcap= myChamber.endcap();
              station = myChamber.station();
                if(endcap==2) station = -station;
              ring = myChamber.ring();
              chamber=myChamber.chamber();

                hResidualRPhiCSC->Fill(residualRPhi);
                hResidualPhiCSC->Fill(residualPhi);
                hResidualZCSC->Fill(residualZ);

                int index=2*station+ring+7;
                if(station==-1) {index=5+ring;
				if(ring==4) index=6;}
                if(station==1) {index=8+ring;
				if(ring==4) index=9;}
                hResidualRPhiCSC_ME[index]->Fill(residualRPhi);
                hResidualPhiCSC_ME[index]->Fill(residualPhi);
                hResidualZCSC_ME[index]->Fill(residualZ);

            }
// Fill individual chamber histograms
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

	//create new histograms

	    char nameOfHistoRPhi[50];
	    char nameOfHistoPhi[50];
	    char nameOfHistoZ[50];
	
	    if(det==1){ // DT
	    sprintf(nameOfHistoRPhi, "ResidualRPhi_W%ldMB%1dS%1d",wheel,station,sector );
	    sprintf(nameOfHistoPhi, "ResidualPhi_W%ldMB%1dS%1d",wheel,station,sector);
	    sprintf(nameOfHistoZ, "ResidualZ_W%ldMB%1dS%1d",wheel,station,sector);
	    }
	    else if(det==2){ //CSC
	    sprintf(nameOfHistoRPhi, "ResidualRPhi_ME%ldR%1dCh%1d",station,ring,chamber );
	    sprintf(nameOfHistoPhi, "ResidualPhi_ME%ldR%1dCh%1d",station,ring,chamber);
	    sprintf(nameOfHistoZ, "ResidualZ_ME%ldR%1dCh%1d",station,ring,chamber);

	    }		    
	    
	    TH1F *histoRPhi = fs->make<TH1F>(nameOfHistoRPhi, nameOfHistoRPhi, 100, -2.0*range, 2.0*range);
	    TH1F *histoPhi = fs->make<TH1F>(nameOfHistoPhi, nameOfHistoPhi, 100, -0.1*range, 0.1*range);
	    TH1F *histoZ = fs->make<TH1F>(nameOfHistoZ, nameOfHistoZ, 100, -2.0*range, 2.0*range);

	    histoRPhi->Fill(residualRPhi);
	    histoPhi->Fill(residualPhi);
	    histoZ->Fill(residualZ);	
	    //Push them into their respective vectors
	    unitsRPhi.push_back(histoRPhi);
	    unitsPhi.push_back(histoPhi);
	    unitsZ.push_back(histoZ);

	    } // new detector
	    else {
	    //If the detector was not new, just fill the histogram
	    unitsRPhi.at(position)->Fill(residualRPhi);
	    unitsPhi.at(position)->Fill(residualPhi);
	    unitsZ.at(position)->Fill(residualZ);
		
	  }
	  countPoints++;
	
 	  innerTSOS = destiny;


	}catch(...) {
	  edm::LogError("MuonAlignmentAnalyzer") <<" Error!! Exception in propagator catched" << endl;
	  continue;
	}

	} //loop over my4DTrack
	} //TSOS was valid

	} // loop over recHitsSize

	numberOfHits=numberOfHits+countPoints;
	} //loop over STAtracks


 } //end doResplots
 

}


/** \class MuonAlignmentAnalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2011/09/04 17:40:58 $
 *  $Revision: 1.13 $
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

#include "Alignment/OfflineValidation/plugins/MuonAlignmentAnalyzer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"

#include "TH2F.h"
#include "TLorentzVector.h"

/// Constructor
MuonAlignmentAnalyzer::MuonAlignmentAnalyzer(const edm::ParameterSet &pset)
    : magFieldToken_(esConsumes()),
      trackingGeometryToken_(esConsumes()),
      theSTAMuonTag(pset.getParameter<edm::InputTag>("StandAloneTrackCollectionTag")),
      theGLBMuonTag(pset.getParameter<edm::InputTag>("GlobalMuonTrackCollectionTag")),
      theRecHits4DTagDT(pset.getParameter<edm::InputTag>("RecHits4DDTCollectionTag")),
      theRecHits4DTagCSC(pset.getParameter<edm::InputTag>("RecHits4DCSCCollectionTag")),
      theDataType(pset.getUntrackedParameter<std::string>("DataType")),
      doSAplots(pset.getUntrackedParameter<bool>("doSAplots")),
      doGBplots(pset.getUntrackedParameter<bool>("doGBplots")),
      doResplots(pset.getUntrackedParameter<bool>("doResplots")),
      ptRangeMin(pset.getUntrackedParameter<double>("ptRangeMin")),
      ptRangeMax(pset.getUntrackedParameter<double>("ptRangeMax")),
      invMassRangeMin(pset.getUntrackedParameter<double>("invMassRangeMin")),
      invMassRangeMax(pset.getUntrackedParameter<double>("invMassRangeMax")),
      resLocalXRangeStation1(pset.getUntrackedParameter<double>("resLocalXRangeStation1")),
      resLocalXRangeStation2(pset.getUntrackedParameter<double>("resLocalXRangeStation2")),
      resLocalXRangeStation3(pset.getUntrackedParameter<double>("resLocalXRangeStation3")),
      resLocalXRangeStation4(pset.getUntrackedParameter<double>("resLocalXRangeStation4")),
      resLocalYRangeStation1(pset.getUntrackedParameter<double>("resLocalYRangeStation1")),
      resLocalYRangeStation2(pset.getUntrackedParameter<double>("resLocalYRangeStation2")),
      resLocalYRangeStation3(pset.getUntrackedParameter<double>("resLocalYRangeStation3")),
      resLocalYRangeStation4(pset.getUntrackedParameter<double>("resLocalYRangeStation4")),
      resPhiRange(pset.getUntrackedParameter<double>("resPhiRange")),
      resThetaRange(pset.getUntrackedParameter<double>("resThetaRange")),
      nbins(pset.getUntrackedParameter<unsigned int>("nbins")),
      min1DTrackRecHitSize(pset.getUntrackedParameter<unsigned int>("min1DTrackRecHitSize")),
      min4DTrackSegmentSize(pset.getUntrackedParameter<unsigned int>("min4DTrackSegmentSize")),
      simTrackToken_(consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"))),
      staTrackToken_(consumes<reco::TrackCollection>(theSTAMuonTag)),
      glbTrackToken_(consumes<reco::TrackCollection>(theGLBMuonTag)),
      allDTSegmentToken_(consumes<DTRecSegment4DCollection>(theRecHits4DTagDT)),
      allCSCSegmentToken_(consumes<CSCSegmentCollection>(theRecHits4DTagCSC)),
      hGBNmuons(nullptr),
      hSANmuons(nullptr),
      hSimNmuons(nullptr),
      hGBNmuons_Barrel(nullptr),
      hSANmuons_Barrel(nullptr),
      hSimNmuons_Barrel(nullptr),
      hGBNmuons_Endcap(nullptr),
      hSANmuons_Endcap(nullptr),
      hSimNmuons_Endcap(nullptr),
      hGBNhits(nullptr),
      hGBNhits_Barrel(nullptr),
      hGBNhits_Endcap(nullptr),
      hSANhits(nullptr),
      hSANhits_Barrel(nullptr),
      hSANhits_Endcap(nullptr),
      hGBChi2(nullptr),
      hSAChi2(nullptr),
      hGBChi2_Barrel(nullptr),
      hSAChi2_Barrel(nullptr),
      hGBChi2_Endcap(nullptr),
      hSAChi2_Endcap(nullptr),
      hGBInvM(nullptr),
      hSAInvM(nullptr),
      hSimInvM(nullptr),
      hGBInvM_Barrel(nullptr),
      hSAInvM_Barrel(nullptr),
      hSimInvM_Barrel(nullptr),
      hGBInvM_Endcap(nullptr),
      hSAInvM_Endcap(nullptr),
      hSimInvM_Endcap(nullptr),
      hGBInvM_Overlap(nullptr),
      hSAInvM_Overlap(nullptr),
      hSimInvM_Overlap(nullptr),
      hSAPTRec(nullptr),
      hGBPTRec(nullptr),
      hSimPT(nullptr),
      hSAPTRec_Barrel(nullptr),
      hGBPTRec_Barrel(nullptr),
      hSimPT_Barrel(nullptr),
      hSAPTRec_Endcap(nullptr),
      hGBPTRec_Endcap(nullptr),
      hSimPT_Endcap(nullptr),
      hGBPTvsEta(nullptr),
      hGBPTvsPhi(nullptr),
      hSAPTvsEta(nullptr),
      hSAPTvsPhi(nullptr),
      hSimPTvsEta(nullptr),
      hSimPTvsPhi(nullptr),
      hSimPhivsEta(nullptr),
      hSAPhivsEta(nullptr),
      hGBPhivsEta(nullptr),
      hSAPTres(nullptr),
      hSAinvPTres(nullptr),
      hGBPTres(nullptr),
      hGBinvPTres(nullptr),
      hSAPTres_Barrel(nullptr),
      hSAPTres_Endcap(nullptr),
      hGBPTres_Barrel(nullptr),
      hGBPTres_Endcap(nullptr),
      hSAPTDiff(nullptr),
      hGBPTDiff(nullptr),
      hSAPTDiffvsEta(nullptr),
      hSAPTDiffvsPhi(nullptr),
      hGBPTDiffvsEta(nullptr),
      hGBPTDiffvsPhi(nullptr),
      hGBinvPTvsEta(nullptr),
      hGBinvPTvsPhi(nullptr),
      hSAinvPTvsEta(nullptr),
      hSAinvPTvsPhi(nullptr),
      hSAinvPTvsNhits(nullptr),
      hGBinvPTvsNhits(nullptr),
      hResidualLocalXDT(nullptr),
      hResidualLocalPhiDT(nullptr),
      hResidualLocalThetaDT(nullptr),
      hResidualLocalYDT(nullptr),
      hResidualLocalXCSC(nullptr),
      hResidualLocalPhiCSC(nullptr),
      hResidualLocalThetaCSC(nullptr),
      hResidualLocalYCSC(nullptr),

      hResidualLocalXDT_W(5),
      hResidualLocalPhiDT_W(5),
      hResidualLocalThetaDT_W(5),
      hResidualLocalYDT_W(5),
      hResidualLocalXCSC_ME(18),
      hResidualLocalPhiCSC_ME(18),
      hResidualLocalThetaCSC_ME(18),
      hResidualLocalYCSC_ME(18),
      hResidualLocalXDT_MB(20),
      hResidualLocalPhiDT_MB(20),
      hResidualLocalThetaDT_MB(20),
      hResidualLocalYDT_MB(20),

      hResidualGlobalRPhiDT(nullptr),
      hResidualGlobalPhiDT(nullptr),
      hResidualGlobalThetaDT(nullptr),
      hResidualGlobalZDT(nullptr),
      hResidualGlobalRPhiCSC(nullptr),
      hResidualGlobalPhiCSC(nullptr),
      hResidualGlobalThetaCSC(nullptr),
      hResidualGlobalRCSC(nullptr),

      hResidualGlobalRPhiDT_W(5),
      hResidualGlobalPhiDT_W(5),
      hResidualGlobalThetaDT_W(5),
      hResidualGlobalZDT_W(5),
      hResidualGlobalRPhiCSC_ME(18),
      hResidualGlobalPhiCSC_ME(18),
      hResidualGlobalThetaCSC_ME(18),
      hResidualGlobalRCSC_ME(18),
      hResidualGlobalRPhiDT_MB(20),
      hResidualGlobalPhiDT_MB(20),
      hResidualGlobalThetaDT_MB(20),
      hResidualGlobalZDT_MB(20),

      hprofLocalPositionCSC(nullptr),
      hprofLocalAngleCSC(nullptr),
      hprofLocalPositionRmsCSC(nullptr),
      hprofLocalAngleRmsCSC(nullptr),
      hprofGlobalPositionCSC(nullptr),
      hprofGlobalAngleCSC(nullptr),
      hprofGlobalPositionRmsCSC(nullptr),
      hprofGlobalAngleRmsCSC(nullptr),
      hprofLocalPositionDT(nullptr),
      hprofLocalAngleDT(nullptr),
      hprofLocalPositionRmsDT(nullptr),
      hprofLocalAngleRmsDT(nullptr),
      hprofGlobalPositionDT(nullptr),
      hprofGlobalAngleDT(nullptr),
      hprofGlobalPositionRmsDT(nullptr),
      hprofGlobalAngleRmsDT(nullptr),
      hprofLocalXDT(nullptr),
      hprofLocalPhiDT(nullptr),
      hprofLocalThetaDT(nullptr),
      hprofLocalYDT(nullptr),
      hprofLocalXCSC(nullptr),
      hprofLocalPhiCSC(nullptr),
      hprofLocalThetaCSC(nullptr),
      hprofLocalYCSC(nullptr),
      hprofGlobalRPhiDT(nullptr),
      hprofGlobalPhiDT(nullptr),
      hprofGlobalThetaDT(nullptr),
      hprofGlobalZDT(nullptr),
      hprofGlobalRPhiCSC(nullptr),
      hprofGlobalPhiCSC(nullptr),
      hprofGlobalThetaCSC(nullptr),
      hprofGlobalRCSC(nullptr) {
  usesResource(TFileService::kSharedResource);

  if (theDataType != "RealData" && theDataType != "SimData")
    edm::LogError("MuonAlignmentAnalyzer") << "Error in Data Type!!" << std::endl;

  numberOfSimTracks = 0;
  numberOfSARecTracks = 0;
  numberOfGBRecTracks = 0;
  numberOfHits = 0;
}

void MuonAlignmentAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("StandAloneTrackCollectionTag", edm::InputTag("globalMuons"));
  desc.add<edm::InputTag>("GlobalMuonTrackCollectionTag", edm::InputTag("standAloneMuons", "UpdatedAtVtx"));
  desc.add<edm::InputTag>("RecHits4DDTCollectionTag", edm::InputTag("dt4DSegments"));
  desc.add<edm::InputTag>("RecHits4DCSCCollectionTag", edm::InputTag("cscSegments"));
  desc.addUntracked<std::string>("DataType", "RealData");
  desc.addUntracked<double>("ptRangeMin", 0.0);
  desc.addUntracked<double>("ptRangeMax", 300.0);
  desc.addUntracked<double>("invMassRangeMin", 0.0);
  desc.addUntracked<double>("invMassRangeMax", 200.0);
  desc.addUntracked<bool>("doSAplots", true);
  desc.addUntracked<bool>("doGBplots", true);
  desc.addUntracked<bool>("doResplots", true);
  desc.addUntracked<double>("resLocalXRangeStation1", 0.1);
  desc.addUntracked<double>("resLocalXRangeStation2", 0.3);
  desc.addUntracked<double>("resLocalXRangeStation3", 3.0);
  desc.addUntracked<double>("resLocalXRangeStation4", 3.0);
  desc.addUntracked<double>("resLocalYRangeStation1", 0.7);
  desc.addUntracked<double>("resLocalYRangeStation2", 0.7);
  desc.addUntracked<double>("resLocalYRangeStation3", 5.0);
  desc.addUntracked<double>("resLocalYRangeStation4", 5.0);
  desc.addUntracked<double>("resThetaRange", 0.1);
  desc.addUntracked<double>("resPhiRange", 0.1);
  desc.addUntracked<int>("nbins", 500);
  desc.addUntracked<int>("min1DTrackRecHitSize", 1);
  desc.addUntracked<int>("min4DTrackSegmentSize", 1);
  descriptions.add("muonAlignmentAnalyzer", desc);
}

void MuonAlignmentAnalyzer::beginJob() {
  //  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  //Create the propagator
  //  if(doResplots)  thePropagator = new SteppingHelixPropagator(&*theMGField, alongMomentum);

  int nBinsPt = (int)fabs(ptRangeMax - ptRangeMin);
  int nBinsMass = (int)fabs(invMassRangeMax - invMassRangeMin);

  // Define and book histograms for SA and GB muon quantities/objects

  if (doGBplots) {
    hGBNmuons = fs->make<TH1F>("GBNmuons", "Nmuons", 10, 0, 10);
    hGBNmuons_Barrel = fs->make<TH1F>("GBNmuons_Barrel", "Nmuons", 10, 0, 10);
    hGBNmuons_Endcap = fs->make<TH1F>("GBNmuons_Endcap", "Nmuons", 10, 0, 10);
    hGBNhits = fs->make<TH1F>("GBNhits", "Nhits", 100, 0, 100);
    hGBNhits_Barrel = fs->make<TH1F>("GBNhits_Barrel", "Nhits", 100, 0, 100);
    hGBNhits_Endcap = fs->make<TH1F>("GBNhits_Endcap", "Nhits", 100, 0, 100);
    hGBPTRec = fs->make<TH1F>("GBpTRec", "p_{T}^{rec}", nBinsPt, ptRangeMin, ptRangeMax);
    hGBPTRec_Barrel = fs->make<TH1F>("GBpTRec_Barrel", "p_{T}^{rec}", nBinsPt, ptRangeMin, ptRangeMax);
    hGBPTRec_Endcap = fs->make<TH1F>("GBpTRec_Endcap", "p_{T}^{rec}", nBinsPt, ptRangeMin, ptRangeMax);
    hGBPTvsEta = fs->make<TH2F>("GBPTvsEta", "p_{T}^{rec} VS #eta", 100, -2.5, 2.5, nBinsPt, ptRangeMin, ptRangeMax);
    hGBPTvsPhi =
        fs->make<TH2F>("GBPTvsPhi", "p_{T}^{rec} VS #phi", 100, -3.1416, 3.1416, nBinsPt, ptRangeMin, ptRangeMax);
    hGBPhivsEta = fs->make<TH2F>("GBPhivsEta", "#phi VS #eta", 100, -2.5, 2.5, 100, -3.1416, 3.1416);

    if (theDataType == "SimData") {
      hGBPTDiff = fs->make<TH1F>("GBpTDiff", "p_{T}^{rec} - p_{T}^{gen} ", 250, -120, 120);
      hGBPTDiffvsEta =
          fs->make<TH2F>("GBPTDiffvsEta", "p_{T}^{rec} - p_{T}^{gen} VS #eta", 100, -2.5, 2.5, 250, -120, 120);
      hGBPTDiffvsPhi =
          fs->make<TH2F>("GBPTDiffvsPhi", "p_{T}^{rec} - p_{T}^{gen} VS #phi", 100, -3.1416, 3.1416, 250, -120, 120);
      hGBPTres = fs->make<TH1F>("GBpTRes", "pT Resolution", 100, -2, 2);
      hGBPTres_Barrel = fs->make<TH1F>("GBpTRes_Barrel", "pT Resolution", 100, -2, 2);
      hGBPTres_Endcap = fs->make<TH1F>("GBpTRes_Endcap", "pT Resolution", 100, -2, 2);
      hGBinvPTres = fs->make<TH1F>("GBinvPTRes", "#sigma (q/p_{T}) Resolution", 100, -2, 2);
      hGBinvPTvsEta = fs->make<TH2F>("GBinvPTvsEta", "#sigma (q/p_{T}) VS #eta", 100, -2.5, 2.5, 100, -2, 2);
      hGBinvPTvsPhi = fs->make<TH2F>("GBinvPTvsPhi", "#sigma (q/p_{T}) VS #phi", 100, -3.1416, 3.1416, 100, -2, 2);
      hGBinvPTvsNhits = fs->make<TH2F>("GBinvPTvsNhits", "#sigma (q/p_{T}) VS Nhits", 100, 0, 100, 100, -2, 2);
    }

    hGBChi2 = fs->make<TH1F>("GBChi2", "Chi2", 200, 0, 200);
    hGBChi2_Barrel = fs->make<TH1F>("GBChi2_Barrel", "Chi2", 200, 0, 200);
    hGBChi2_Endcap = fs->make<TH1F>("GBChi2_Endcap ", "Chi2", 200, 0, 200);
    hGBInvM = fs->make<TH1F>("GBInvM", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hGBInvM_Barrel = fs->make<TH1F>("GBInvM_Barrel", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hGBInvM_Endcap = fs->make<TH1F>("GBInvM_Endcap ", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hGBInvM_Overlap = fs->make<TH1F>("GBInvM_Overlap", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
  }

  if (doSAplots) {
    hSANmuons = fs->make<TH1F>("SANmuons", "Nmuons", 10, 0, 10);
    hSANmuons_Barrel = fs->make<TH1F>("SANmuons_Barrel", "Nmuons", 10, 0, 10);
    hSANmuons_Endcap = fs->make<TH1F>("SANmuons_Endcap", "Nmuons", 10, 0, 10);
    hSANhits = fs->make<TH1F>("SANhits", "Nhits", 100, 0, 100);
    hSANhits_Barrel = fs->make<TH1F>("SANhits_Barrel", "Nhits", 100, 0, 100);
    hSANhits_Endcap = fs->make<TH1F>("SANhits_Endcap", "Nhits", 100, 0, 100);
    hSAPTRec = fs->make<TH1F>("SApTRec", "p_{T}^{rec}", nBinsPt, ptRangeMin, ptRangeMax);
    hSAPTRec_Barrel = fs->make<TH1F>("SApTRec_Barrel", "p_{T}^{rec}", nBinsPt, ptRangeMin, ptRangeMax);
    hSAPTRec_Endcap = fs->make<TH1F>("SApTRec_Endcap", "p_{T}^{rec}", nBinsPt, ptRangeMin, ptRangeMax);
    hSAPTvsEta = fs->make<TH2F>("SAPTvsEta", "p_{T}^{rec} VS #eta", 100, -2.5, 2.5, nBinsPt, ptRangeMin, ptRangeMax);
    hSAPTvsPhi =
        fs->make<TH2F>("SAPTvsPhi", "p_{T}^{rec} VS #phi", 100, -3.1416, 3.1416, nBinsPt, ptRangeMin, ptRangeMax);
    hSAPhivsEta = fs->make<TH2F>("SAPhivsEta", "#phi VS #eta", 100, -2.5, 2.5, 100, -3.1416, 3.1416);

    if (theDataType == "SimData") {
      hSAPTDiff = fs->make<TH1F>("SApTDiff", "p_{T}^{rec} - p_{T}^{gen} ", 250, -120, 120);
      hSAPTDiffvsEta =
          fs->make<TH2F>("SAPTDiffvsEta", "p_{T}^{rec} - p_{T}^{gen} VS #eta", 100, -2.5, 2.5, 250, -120, 120);
      hSAPTDiffvsPhi =
          fs->make<TH2F>("SAPTDiffvsPhi", "p_{T}^{rec} - p_{T}^{gen} VS #phi", 100, -3.1416, 3.1416, 250, -120, 120);
      hSAPTres = fs->make<TH1F>("SApTRes", "pT Resolution", 100, -2, 2);
      hSAPTres_Barrel = fs->make<TH1F>("SApTRes_Barrel", "pT Resolution", 100, -2, 2);
      hSAPTres_Endcap = fs->make<TH1F>("SApTRes_Endcap", "pT Resolution", 100, -2, 2);
      hSAinvPTres = fs->make<TH1F>("SAinvPTRes", "1/pT Resolution", 100, -2, 2);

      hSAinvPTvsEta = fs->make<TH2F>("SAinvPTvsEta", "#sigma (q/p_{T}) VS #eta", 100, -2.5, 2.5, 100, -2, 2);
      hSAinvPTvsPhi = fs->make<TH2F>("SAinvPTvsPhi", "#sigma (q/p_{T}) VS #phi", 100, -3.1416, 3.1416, 100, -2, 2);
      hSAinvPTvsNhits = fs->make<TH2F>("SAinvPTvsNhits", "#sigma (q/p_{T}) VS Nhits", 100, 0, 100, 100, -2, 2);
    }
    hSAInvM = fs->make<TH1F>("SAInvM", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSAInvM_Barrel = fs->make<TH1F>("SAInvM_Barrel", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSAInvM_Endcap = fs->make<TH1F>("SAInvM_Endcap", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSAInvM_Overlap = fs->make<TH1F>("SAInvM_Overlap", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSAChi2 = fs->make<TH1F>("SAChi2", "Chi2", 200, 0, 200);
    hSAChi2_Barrel = fs->make<TH1F>("SAChi2_Barrel", "Chi2", 200, 0, 200);
    hSAChi2_Endcap = fs->make<TH1F>("SAChi2_Endcap", "Chi2", 200, 0, 200);
  }

  if (theDataType == "SimData") {
    hSimNmuons = fs->make<TH1F>("SimNmuons", "Nmuons", 10, 0, 10);
    hSimNmuons_Barrel = fs->make<TH1F>("SimNmuons_Barrel", "Nmuons", 10, 0, 10);
    hSimNmuons_Endcap = fs->make<TH1F>("SimNmuons_Endcap", "Nmuons", 10, 0, 10);
    hSimPT = fs->make<TH1F>("SimPT", "p_{T}^{gen} ", nBinsPt, ptRangeMin, ptRangeMax);
    hSimPT_Barrel = fs->make<TH1F>("SimPT_Barrel", "p_{T}^{gen} ", nBinsPt, ptRangeMin, ptRangeMax);
    hSimPT_Endcap = fs->make<TH1F>("SimPT_Endcap", "p_{T}^{gen} ", nBinsPt, ptRangeMin, ptRangeMax);
    hSimPTvsEta = fs->make<TH2F>("SimPTvsEta", "p_{T}^{gen} VS #eta", 100, -2.5, 2.5, nBinsPt, ptRangeMin, ptRangeMax);
    hSimPTvsPhi =
        fs->make<TH2F>("SimPTvsPhi", "p_{T}^{gen} VS #phi", 100, -3.1416, 3.1416, nBinsPt, ptRangeMin, ptRangeMax);
    hSimPhivsEta = fs->make<TH2F>("SimPhivsEta", "#phi VS #eta", 100, -2.5, 2.5, 100, -3.1416, 3.1416);
    hSimInvM = fs->make<TH1F>("SimInvM", "M_{inv}^{gen} ", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSimInvM_Barrel = fs->make<TH1F>("SimInvM_Barrel", "M_{inv}^{rec}", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSimInvM_Endcap = fs->make<TH1F>("SimInvM_Endcap", "M_{inv}^{gen} ", nBinsMass, invMassRangeMin, invMassRangeMax);
    hSimInvM_Overlap = fs->make<TH1F>("SimInvM_Overlap", "M_{inv}^{gen} ", nBinsMass, invMassRangeMin, invMassRangeMax);
  }

  if (doResplots) {
    // All DT and CSC chambers
    hResidualLocalXDT = fs->make<TH1F>("hResidualLocalXDT", "hResidualLocalXDT", 200, -10, 10);
    hResidualLocalPhiDT = fs->make<TH1F>("hResidualLocalPhiDT", "hResidualLocalPhiDT", 100, -1, 1);
    hResidualLocalThetaDT = fs->make<TH1F>("hResidualLocalThetaDT", "hResidualLocalThetaDT", 100, -1, 1);
    hResidualLocalYDT = fs->make<TH1F>("hResidualLocalYDT", "hResidualLocalYDT", 200, -10, 10);
    hResidualLocalXCSC = fs->make<TH1F>("hResidualLocalXCSC", "hResidualLocalXCSC", 200, -10, 10);
    hResidualLocalPhiCSC = fs->make<TH1F>("hResidualLocalPhiCSC", "hResidualLocalPhiCSC", 100, -1, 1);
    hResidualLocalThetaCSC = fs->make<TH1F>("hResidualLocalThetaCSC", "hResidualLocalThetaCSC", 100, -1, 1);
    hResidualLocalYCSC = fs->make<TH1F>("hResidualLocalYCSC", "hResidualLocalYCSC", 200, -10, 10);
    hResidualGlobalRPhiDT = fs->make<TH1F>("hResidualGlobalRPhiDT", "hResidualGlobalRPhiDT", 200, -10, 10);
    hResidualGlobalPhiDT = fs->make<TH1F>("hResidualGlobalPhiDT", "hResidualGlobalPhiDT", 100, -1, 1);
    hResidualGlobalThetaDT = fs->make<TH1F>("hResidualGlobalThetaDT", "hResidualGlobalThetaDT", 100, -1, 1);
    hResidualGlobalZDT = fs->make<TH1F>("hResidualGlobalZDT", "hResidualGlobalZDT", 200, -10, 10);
    hResidualGlobalRPhiCSC = fs->make<TH1F>("hResidualGlobalRPhiCSC", "hResidualGlobalRPhiCSC", 200, -10, 10);
    hResidualGlobalPhiCSC = fs->make<TH1F>("hResidualGlobalPhiCSC", "hResidualGlobalPhiCSC", 100, -1, 1);
    hResidualGlobalThetaCSC = fs->make<TH1F>("hResidualGlobalThetaCSC", "hResidualGlobalThetaCSC", 100, -1, 1);
    hResidualGlobalRCSC = fs->make<TH1F>("hResidualGlobalRCSC", "hResidualGlobalRCSC", 200, -10, 10);

    // DT Wheels
    hResidualLocalXDT_W[0] = fs->make<TH1F>("hResidualLocalXDT_W-2", "hResidualLocalXDT_W-2", 200, -10, 10);
    hResidualLocalPhiDT_W[0] = fs->make<TH1F>("hResidualLocalPhiDT_W-2", "hResidualLocalPhiDT_W-2", 200, -1, 1);
    hResidualLocalThetaDT_W[0] = fs->make<TH1F>("hResidualLocalThetaDT_W-2", "hResidualLocalThetaDT_W-2", 200, -1, 1);
    hResidualLocalYDT_W[0] = fs->make<TH1F>("hResidualLocalYDT_W-2", "hResidualLocalYDT_W-2", 200, -10, 10);
    hResidualLocalXDT_W[1] = fs->make<TH1F>("hResidualLocalXDT_W-1", "hResidualLocalXDT_W-1", 200, -10, 10);
    hResidualLocalPhiDT_W[1] = fs->make<TH1F>("hResidualLocalPhiDT_W-1", "hResidualLocalPhiDT_W-1", 200, -1, 1);
    hResidualLocalThetaDT_W[1] = fs->make<TH1F>("hResidualLocalThetaDT_W-1", "hResidualLocalThetaDT_W-1", 200, -1, 1);
    hResidualLocalYDT_W[1] = fs->make<TH1F>("hResidualLocalYDT_W-1", "hResidualLocalYDT_W-1", 200, -10, 10);
    hResidualLocalXDT_W[2] = fs->make<TH1F>("hResidualLocalXDT_W0", "hResidualLocalXDT_W0", 200, -10, 10);
    hResidualLocalPhiDT_W[2] = fs->make<TH1F>("hResidualLocalPhiDT_W0", "hResidualLocalPhiDT_W0", 200, -1, 1);
    hResidualLocalThetaDT_W[2] = fs->make<TH1F>("hResidualLocalThetaDT_W0", "hResidualLocalThetaDT_W0", 200, -1, 1);
    hResidualLocalYDT_W[2] = fs->make<TH1F>("hResidualLocalYDT_W0", "hResidualLocalYDT_W0", 200, -10, 10);
    hResidualLocalXDT_W[3] = fs->make<TH1F>("hResidualLocalXDT_W1", "hResidualLocalXDT_W1", 200, -10, 10);
    hResidualLocalPhiDT_W[3] = fs->make<TH1F>("hResidualLocalPhiDT_W1", "hResidualLocalPhiDT_W1", 200, -1, 1);
    hResidualLocalThetaDT_W[3] = fs->make<TH1F>("hResidualLocalThetaDT_W1", "hResidualLocalThetaDT_W1", 200, -1, 1);
    hResidualLocalYDT_W[3] = fs->make<TH1F>("hResidualLocalYDT_W1", "hResidualLocalYDT_W1", 200, -10, 10);
    hResidualLocalXDT_W[4] = fs->make<TH1F>("hResidualLocalXDT_W2", "hResidualLocalXDT_W2", 200, -10, 10);
    hResidualLocalPhiDT_W[4] = fs->make<TH1F>("hResidualLocalPhiDT_W2", "hResidualLocalPhiDT_W2", 200, -1, 1);
    hResidualLocalThetaDT_W[4] = fs->make<TH1F>("hResidualLocalThetaDT_W2", "hResidualLocalThetaDT_W2", 200, -1, 1);
    hResidualLocalYDT_W[4] = fs->make<TH1F>("hResidualLocalYDT_W2", "hResidualLocalYDT_W2", 200, -10, 10);
    hResidualGlobalRPhiDT_W[0] = fs->make<TH1F>("hResidualGlobalRPhiDT_W-2", "hResidualGlobalRPhiDT_W-2", 200, -10, 10);
    hResidualGlobalPhiDT_W[0] = fs->make<TH1F>("hResidualGlobalPhiDT_W-2", "hResidualGlobalPhiDT_W-2", 200, -1, 1);
    hResidualGlobalThetaDT_W[0] =
        fs->make<TH1F>("hResidualGlobalThetaDT_W-2", "hResidualGlobalThetaDT_W-2", 200, -1, 1);
    hResidualGlobalZDT_W[0] = fs->make<TH1F>("hResidualGlobalZDT_W-2", "hResidualGlobalZDT_W-2", 200, -10, 10);
    hResidualGlobalRPhiDT_W[1] = fs->make<TH1F>("hResidualGlobalRPhiDT_W-1", "hResidualGlobalRPhiDT_W-1", 200, -10, 10);
    hResidualGlobalPhiDT_W[1] = fs->make<TH1F>("hResidualGlobalPhiDT_W-1", "hResidualGlobalPhiDT_W-1", 200, -1, 1);
    hResidualGlobalThetaDT_W[1] =
        fs->make<TH1F>("hResidualGlobalThetaDT_W-1", "hResidualGlobalThetaDT_W-1", 200, -1, 1);
    hResidualGlobalZDT_W[1] = fs->make<TH1F>("hResidualGlobalZDT_W-1", "hResidualGlobalZDT_W-1", 200, -10, 10);
    hResidualGlobalRPhiDT_W[2] = fs->make<TH1F>("hResidualGlobalRPhiDT_W0", "hResidualGlobalRPhiDT_W0", 200, -10, 10);
    hResidualGlobalPhiDT_W[2] = fs->make<TH1F>("hResidualGlobalPhiDT_W0", "hResidualGlobalPhiDT_W0", 200, -1, 1);
    hResidualGlobalThetaDT_W[2] = fs->make<TH1F>("hResidualGlobalThetaDT_W0", "hResidualGlobalThetaDT_W0", 200, -1, 1);
    hResidualGlobalZDT_W[2] = fs->make<TH1F>("hResidualGlobalZDT_W0", "hResidualGlobalZDT_W0", 200, -10, 10);
    hResidualGlobalRPhiDT_W[3] = fs->make<TH1F>("hResidualGlobalRPhiDT_W1", "hResidualGlobalRPhiDT_W1", 200, -10, 10);
    hResidualGlobalPhiDT_W[3] = fs->make<TH1F>("hResidualGlobalPhiDT_W1", "hResidualGlobalPhiDT_W1", 200, -1, 1);
    hResidualGlobalThetaDT_W[3] = fs->make<TH1F>("hResidualGlobalThetaDT_W1", "hResidualGlobalThetaDT_W1", 200, -1, 1);
    hResidualGlobalZDT_W[3] = fs->make<TH1F>("hResidualGlobalZDT_W1", "hResidualGlobalZDT_W1", 200, -10, 10);
    hResidualGlobalRPhiDT_W[4] = fs->make<TH1F>("hResidualGlobalRPhiDT_W2", "hResidualGlobalRPhiDT_W2", 200, -10, 10);
    hResidualGlobalPhiDT_W[4] = fs->make<TH1F>("hResidualGlobalPhiDT_W2", "hResidualGlobalPhiDT_W2", 200, -1, 1);
    hResidualGlobalThetaDT_W[4] = fs->make<TH1F>("hResidualGlobalThetaDT_W2", "hResidualGlobalThetaDT_W2", 200, -1, 1);
    hResidualGlobalZDT_W[4] = fs->make<TH1F>("hResidualGlobalZDT_W2", "hResidualGlobalZDT_W2", 200, -10, 10);

    // DT Stations
    hResidualLocalXDT_MB[0] = fs->make<TH1F>("hResidualLocalXDT_MB-2/1", "hResidualLocalXDT_MB-2/1", 200, -10, 10);
    hResidualLocalPhiDT_MB[0] = fs->make<TH1F>("hResidualLocalPhiDT_MB-2/1", "hResidualLocalPhiDT_MB-2/1", 200, -1, 1);
    hResidualLocalThetaDT_MB[0] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-2/1", "hResidualLocalThetaDT_MB-2/1", 200, -1, 1);
    hResidualLocalYDT_MB[0] = fs->make<TH1F>("hResidualLocalYDT_MB-2/1", "hResidualLocalYDT_MB-2/1", 200, -10, 10);
    hResidualLocalXDT_MB[1] = fs->make<TH1F>("hResidualLocalXDT_MB-2/2", "hResidualLocalXDT_MB-2/2", 200, -10, 10);
    hResidualLocalPhiDT_MB[1] = fs->make<TH1F>("hResidualLocalPhiDT_MB-2/2", "hResidualLocalPhiDT_MB-2/2", 200, -1, 1);
    hResidualLocalThetaDT_MB[1] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-2/2", "hResidualLocalThetaDT_MB-2/2", 200, -1, 1);
    hResidualLocalYDT_MB[1] = fs->make<TH1F>("hResidualLocalYDT_MB-2/2", "hResidualLocalYDT_MB-2/2", 200, -10, 10);
    hResidualLocalXDT_MB[2] = fs->make<TH1F>("hResidualLocalXDT_MB-2/3", "hResidualLocalXDT_MB-2/3", 200, -10, 10);
    hResidualLocalPhiDT_MB[2] = fs->make<TH1F>("hResidualLocalPhiDT_MB-2/3", "hResidualLocalPhiDT_MB-2/3", 200, -1, 1);
    hResidualLocalThetaDT_MB[2] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-2/3", "hResidualLocalThetaDT_MB-2/3", 200, -1, 1);
    hResidualLocalYDT_MB[2] = fs->make<TH1F>("hResidualLocalYDT_MB-2/3", "hResidualLocalYDT_MB-2/3", 200, -10, 10);
    hResidualLocalXDT_MB[3] = fs->make<TH1F>("hResidualLocalXDT_MB-2/4", "hResidualLocalXDT_MB-2/4", 200, -10, 10);
    hResidualLocalPhiDT_MB[3] = fs->make<TH1F>("hResidualLocalPhiDT_MB-2/4", "hResidualLocalPhiDT_MB-2/4", 200, -1, 1);
    hResidualLocalThetaDT_MB[3] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-2/4", "hResidualLocalThetaDT_MB-2/4", 200, -1, 1);
    hResidualLocalYDT_MB[3] = fs->make<TH1F>("hResidualLocalYDT_MB-2/4", "hResidualLocalYDT_MB-2/4", 200, -10, 10);
    hResidualLocalXDT_MB[4] = fs->make<TH1F>("hResidualLocalXDT_MB-1/1", "hResidualLocalXDT_MB-1/1", 200, -10, 10);
    hResidualLocalPhiDT_MB[4] = fs->make<TH1F>("hResidualLocalPhiDT_MB-1/1", "hResidualLocalPhiDT_MB-1/1", 200, -1, 1);
    hResidualLocalThetaDT_MB[4] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-1/1", "hResidualLocalThetaDT_MB-1/1", 200, -1, 1);
    hResidualLocalYDT_MB[4] = fs->make<TH1F>("hResidualLocalYDT_MB-1/1", "hResidualLocalYDT_MB-1/1", 200, -10, 10);
    hResidualLocalXDT_MB[5] = fs->make<TH1F>("hResidualLocalXDT_MB-1/2", "hResidualLocalXDT_MB-1/2", 200, -10, 10);
    hResidualLocalPhiDT_MB[5] = fs->make<TH1F>("hResidualLocalPhiDT_MB-1/2", "hResidualLocalPhiDT_MB-1/2", 200, -1, 1);
    hResidualLocalThetaDT_MB[5] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-1/2", "hResidualLocalThetaDT_MB-1/2", 200, -1, 1);
    hResidualLocalYDT_MB[5] = fs->make<TH1F>("hResidualLocalYDT_MB-1/2", "hResidualLocalYDT_MB-1/2", 200, -10, 10);
    hResidualLocalXDT_MB[6] = fs->make<TH1F>("hResidualLocalXDT_MB-1/3", "hResidualLocalXDT_MB-1/3", 200, -10, 10);
    hResidualLocalPhiDT_MB[6] = fs->make<TH1F>("hResidualLocalPhiDT_MB-1/3", "hResidualLocalPhiDT_MB-1/3", 200, -1, 1);
    hResidualLocalThetaDT_MB[6] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-1/3", "hResidualLocalThetaDT_MB-1/3", 200, -1, 1);
    hResidualLocalYDT_MB[6] = fs->make<TH1F>("hResidualLocalYDT_MB-1/3", "hResidualLocalYDT_MB-1/3", 200, -10, 10);
    hResidualLocalXDT_MB[7] = fs->make<TH1F>("hResidualLocalXDT_MB-1/4", "hResidualLocalXDT_MB-1/4", 200, -10, 10);
    hResidualLocalPhiDT_MB[7] = fs->make<TH1F>("hResidualLocalPhiDT_MB-1/4", "hResidualLocalPhiDT_MB-1/4", 200, -1, 1);
    hResidualLocalThetaDT_MB[7] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB-1/4", "hResidualLocalThetaDT_MB-1/4", 200, -1, 1);
    hResidualLocalYDT_MB[7] = fs->make<TH1F>("hResidualLocalYDT_MB-1/4", "hResidualLocalYDT_MB-1/4", 200, -10, 10);
    hResidualLocalXDT_MB[8] = fs->make<TH1F>("hResidualLocalXDT_MB0/1", "hResidualLocalXDT_MB0/1", 200, -10, 10);
    hResidualLocalPhiDT_MB[8] = fs->make<TH1F>("hResidualLocalPhiDT_MB0/1", "hResidualLocalPhiDT_MB0/1", 200, -1, 1);
    hResidualLocalThetaDT_MB[8] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB0/1", "hResidualLocalThetaDT_MB0/1", 200, -1, 1);
    hResidualLocalYDT_MB[8] = fs->make<TH1F>("hResidualLocalYDT_MB0/1", "hResidualLocalYDT_MB0/1", 200, -10, 10);
    hResidualLocalXDT_MB[9] = fs->make<TH1F>("hResidualLocalXDT_MB0/2", "hResidualLocalXDT_MB0/2", 200, -10, 10);
    hResidualLocalPhiDT_MB[9] = fs->make<TH1F>("hResidualLocalPhiDT_MB0/2", "hResidualLocalPhiDT_MB0/2", 200, -1, 1);
    hResidualLocalThetaDT_MB[9] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB0/2", "hResidualLocalThetaDT_MB0/2", 200, -1, 1);
    hResidualLocalYDT_MB[9] = fs->make<TH1F>("hResidualLocalYDT_MB0/2", "hResidualLocalYDT_MB0/2", 200, -10, 10);
    hResidualLocalXDT_MB[10] = fs->make<TH1F>("hResidualLocalXDT_MB0/3", "hResidualLocalXDT_MB0/3", 200, -10, 10);
    hResidualLocalThetaDT_MB[10] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB0/3", "hResidualLocalThetaDT_MB0/3", 200, -1, 1);
    hResidualLocalPhiDT_MB[10] = fs->make<TH1F>("hResidualLocalPhiDT_MB0/3", "hResidualLocalPhiDT_MB0/3", 200, -1, 1);
    hResidualLocalYDT_MB[10] = fs->make<TH1F>("hResidualLocalYDT_MB0/3", "hResidualLocalYDT_MB0/3", 200, -10, 10);
    hResidualLocalXDT_MB[11] = fs->make<TH1F>("hResidualLocalXDT_MB0/4", "hResidualLocalXDT_MB0/4", 200, -10, 10);
    hResidualLocalPhiDT_MB[11] = fs->make<TH1F>("hResidualLocalPhiDT_MB0/4", "hResidualLocalPhiDT_MB0/4", 200, -1, 1);
    hResidualLocalThetaDT_MB[11] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB0/4", "hResidualLocalThetaDT_MB0/4", 200, -1, 1);
    hResidualLocalYDT_MB[11] = fs->make<TH1F>("hResidualLocalYDT_MB0/4", "hResidualLocalYDT_MB0/4", 200, -10, 10);
    hResidualLocalXDT_MB[12] = fs->make<TH1F>("hResidualLocalXDT_MB1/1", "hResidualLocalXDT_MB1/1", 200, -10, 10);
    hResidualLocalPhiDT_MB[12] = fs->make<TH1F>("hResidualLocalPhiDT_MB1/1", "hResidualLocalPhiDT_MB1/1", 200, -1, 1);
    hResidualLocalThetaDT_MB[12] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB1/1", "hResidualLocalThetaDT_MB1/1", 200, -1, 1);
    hResidualLocalYDT_MB[12] = fs->make<TH1F>("hResidualLocalYDT_MB1/1", "hResidualLocalYDT_MB1/1", 200, -10, 10);
    hResidualLocalXDT_MB[13] = fs->make<TH1F>("hResidualLocalXDT_MB1/2", "hResidualLocalXDT_MB1/2", 200, -10, 10);
    hResidualLocalPhiDT_MB[13] = fs->make<TH1F>("hResidualLocalPhiDT_MB1/2", "hResidualLocalPhiDT_MB1/2", 200, -1, 1);
    hResidualLocalThetaDT_MB[13] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB1/2", "hResidualLocalThetaDT_MB1/2", 200, -1, 1);
    hResidualLocalYDT_MB[13] = fs->make<TH1F>("hResidualLocalYDT_MB1/2", "hResidualLocalYDT_MB1/2", 200, -10, 10);
    hResidualLocalXDT_MB[14] = fs->make<TH1F>("hResidualLocalXDT_MB1/3", "hResidualLocalXDT_MB1/3", 200, -10, 10);
    hResidualLocalPhiDT_MB[14] = fs->make<TH1F>("hResidualLocalPhiDT_MB1/3", "hResidualLocalPhiDT_MB1/3", 200, -1, 1);
    hResidualLocalThetaDT_MB[14] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB1/3", "hResidualLocalThetaDT_MB1/3", 200, -1, 1);
    hResidualLocalYDT_MB[14] = fs->make<TH1F>("hResidualLocalYDT_MB1/3", "hResidualLocalYDT_MB1/3", 200, -10, 10);
    hResidualLocalXDT_MB[15] = fs->make<TH1F>("hResidualLocalXDT_MB1/4", "hResidualLocalXDT_MB1/4", 200, -10, 10);
    hResidualLocalPhiDT_MB[15] = fs->make<TH1F>("hResidualLocalPhiDT_MB1/4", "hResidualLocalPhiDT_MB1/4", 200, -1, 1);
    hResidualLocalThetaDT_MB[15] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB1/4", "hResidualLocalThetaDT_MB1/4", 200, -1, 1);
    hResidualLocalYDT_MB[15] = fs->make<TH1F>("hResidualLocalYDT_MB1/4", "hResidualLocalYDT_MB1/4", 200, -10, 10);
    hResidualLocalXDT_MB[16] = fs->make<TH1F>("hResidualLocalXDT_MB2/1", "hResidualLocalXDT_MB2/1", 200, -10, 10);
    hResidualLocalPhiDT_MB[16] = fs->make<TH1F>("hResidualLocalPhiDT_MB2/1", "hResidualLocalPhiDT_MB2/1", 200, -1, 1);
    hResidualLocalThetaDT_MB[16] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB2/1", "hResidualLocalThetaDT_MB2/1", 200, -1, 1);
    hResidualLocalYDT_MB[16] = fs->make<TH1F>("hResidualLocalYDT_MB2/1", "hResidualLocalYDT_MB2/1", 200, -10, 10);
    hResidualLocalXDT_MB[17] = fs->make<TH1F>("hResidualLocalXDT_MB2/2", "hResidualLocalXDT_MB2/2", 200, -10, 10);
    hResidualLocalPhiDT_MB[17] = fs->make<TH1F>("hResidualLocalPhiDT_MB2/2", "hResidualLocalPhiDT_MB2/2", 200, -1, 1);
    hResidualLocalThetaDT_MB[17] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB2/2", "hResidualLocalThetaDT_MB2/2", 200, -1, 1);
    hResidualLocalYDT_MB[17] = fs->make<TH1F>("hResidualLocalYDT_MB2/2", "hResidualLocalYDT_MB2/2", 200, -10, 10);
    hResidualLocalXDT_MB[18] = fs->make<TH1F>("hResidualLocalXDT_MB2/3", "hResidualLocalXDT_MB2/3", 200, -10, 10);
    hResidualLocalPhiDT_MB[18] = fs->make<TH1F>("hResidualLocalPhiDT_MB2/3", "hResidualLocalPhiDT_MB2/3", 200, -1, 1);
    hResidualLocalThetaDT_MB[18] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB2/3", "hResidualLocalThetaDT_MB2/3", 200, -1, 1);
    hResidualLocalYDT_MB[18] = fs->make<TH1F>("hResidualLocalYDT_MB2/3", "hResidualLocalYDT_MB2/3", 200, -10, 10);
    hResidualLocalXDT_MB[19] = fs->make<TH1F>("hResidualLocalXDT_MB2/4", "hResidualLocalXDT_MB2/4", 200, -10, 10);
    hResidualLocalPhiDT_MB[19] = fs->make<TH1F>("hResidualLocalPhiDT_MB2/4", "hResidualLocalPhiDT_MB2/4", 200, -1, 1);
    hResidualLocalThetaDT_MB[19] =
        fs->make<TH1F>("hResidualLocalThetaDT_MB2/4", "hResidualLocalThetaDT_MB2/4", 200, -1, 1);
    hResidualLocalYDT_MB[19] = fs->make<TH1F>("hResidualLocalYDT_MB2/4", "hResidualLocalYDT_MB2/4", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[0] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-2/1", "hResidualGlobalRPhiDT_MB-2/1", 200, -10, 10);
    hResidualGlobalPhiDT_MB[0] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-2/1", "hResidualGlobalPhiDT_MB-2/1", 200, -1, 1);
    hResidualGlobalThetaDT_MB[0] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-2/1", "hResidualGlobalThetaDT_MB-2/1", 200, -1, 1);
    hResidualGlobalZDT_MB[0] = fs->make<TH1F>("hResidualGlobalZDT_MB-2/1", "hResidualGlobalZDT_MB-2/1", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[1] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-2/2", "hResidualGlobalRPhiDT_MB-2/2", 200, -10, 10);
    hResidualGlobalPhiDT_MB[1] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-2/2", "hResidualGlobalPhiDT_MB-2/2", 200, -1, 1);
    hResidualGlobalThetaDT_MB[1] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-2/2", "hResidualGlobalThetaDT_MB-2/2", 200, -1, 1);
    hResidualGlobalZDT_MB[1] = fs->make<TH1F>("hResidualGlobalZDT_MB-2/2", "hResidualGlobalZDT_MB-2/2", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[2] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-2/3", "hResidualGlobalRPhiDT_MB-2/3", 200, -10, 10);
    hResidualGlobalPhiDT_MB[2] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-2/3", "hResidualGlobalPhiDT_MB-2/3", 200, -1, 1);
    hResidualGlobalThetaDT_MB[2] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-2/3", "hResidualGlobalThetaDT_MB-2/3", 200, -1, 1);
    hResidualGlobalZDT_MB[2] = fs->make<TH1F>("hResidualGlobalZDT_MB-2/3", "hResidualGlobalZDT_MB-2/3", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[3] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-2/4", "hResidualGlobalRPhiDT_MB-2/4", 200, -10, 10);
    hResidualGlobalPhiDT_MB[3] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-2/4", "hResidualGlobalPhiDT_MB-2/4", 200, -1, 1);
    hResidualGlobalThetaDT_MB[3] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-2/4", "hResidualGlobalThetaDT_MB-2/4", 200, -1, 1);
    hResidualGlobalZDT_MB[3] = fs->make<TH1F>("hResidualGlobalZDT_MB-2/4", "hResidualGlobalZDT_MB-2/4", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[4] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-1/1", "hResidualGlobalRPhiDT_MB-1/1", 200, -10, 10);
    hResidualGlobalPhiDT_MB[4] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-1/1", "hResidualGlobalPhiDT_MB-1/1", 200, -1, 1);
    hResidualGlobalThetaDT_MB[4] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-1/1", "hResidualGlobalThetaDT_MB-1/1", 200, -1, 1);
    hResidualGlobalZDT_MB[4] = fs->make<TH1F>("hResidualGlobalZDT_MB-1/1", "hResidualGlobalZDT_MB-1/1", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[5] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-1/2", "hResidualGlobalRPhiDT_MB-1/2", 200, -10, 10);
    hResidualGlobalPhiDT_MB[5] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-1/2", "hResidualGlobalPhiDT_MB-1/2", 200, -1, 1);
    hResidualGlobalThetaDT_MB[5] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-1/2", "hResidualGlobalThetaDT_MB-1/2", 200, -1, 1);
    hResidualGlobalZDT_MB[5] = fs->make<TH1F>("hResidualGlobalZDT_MB-1/2", "hResidualGlobalZDT_MB-1/2", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[6] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-1/3", "hResidualGlobalRPhiDT_MB-1/3", 200, -10, 10);
    hResidualGlobalPhiDT_MB[6] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-1/3", "hResidualGlobalPhiDT_MB-1/3", 200, -1, 1);
    hResidualGlobalThetaDT_MB[6] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-1/3", "hResidualGlobalThetaDT_MB-1/3", 200, -1, 1);
    hResidualGlobalZDT_MB[6] = fs->make<TH1F>("hResidualGlobalZDT_MB-1/3", "hResidualGlobalZDT_MB-1/3", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[7] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB-1/4", "hResidualGlobalRPhiDT_MB-1/4", 200, -10, 10);
    hResidualGlobalPhiDT_MB[7] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB-1/4", "hResidualGlobalPhiDT_MB-1/4", 200, -1, 1);
    hResidualGlobalThetaDT_MB[7] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB-1/4", "hResidualGlobalThetaDT_MB-1/4", 200, -1, 1);
    hResidualGlobalZDT_MB[7] = fs->make<TH1F>("hResidualGlobalZDT_MB-1/4", "hResidualGlobalZDT_MB-1/4", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[8] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB0/1", "hResidualGlobalRPhiDT_MB0/1", 200, -10, 10);
    hResidualGlobalPhiDT_MB[8] = fs->make<TH1F>("hResidualGlobalPhiDT_MB0/1", "hResidualGlobalPhiDT_MB0/1", 200, -1, 1);
    hResidualGlobalThetaDT_MB[8] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB0/1", "hResidualGlobalThetaDT_MB0/1", 200, -1, 1);
    hResidualGlobalZDT_MB[8] = fs->make<TH1F>("hResidualGlobalZDT_MB0/1", "hResidualGlobalZDT_MB0/1", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[9] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB0/2", "hResidualGlobalRPhiDT_MB0/2", 200, -10, 10);
    hResidualGlobalPhiDT_MB[9] = fs->make<TH1F>("hResidualGlobalPhiDT_MB0/2", "hResidualGlobalPhiDT_MB0/2", 200, -1, 1);
    hResidualGlobalThetaDT_MB[9] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB0/2", "hResidualGlobalThetaDT_MB0/2", 200, -1, 1);
    hResidualGlobalZDT_MB[9] = fs->make<TH1F>("hResidualGlobalZDT_MB0/2", "hResidualGlobalZDT_MB0/2", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[10] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB0/3", "hResidualGlobalRPhiDT_MB0/3", 200, -10, 10);
    hResidualGlobalThetaDT_MB[10] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB0/3", "hResidualGlobalThetaDT_MB0/3", 200, -1, 1);
    hResidualGlobalPhiDT_MB[10] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB0/3", "hResidualGlobalPhiDT_MB0/3", 200, -1, 1);
    hResidualGlobalZDT_MB[10] = fs->make<TH1F>("hResidualGlobalZDT_MB0/3", "hResidualGlobalZDT_MB0/3", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[11] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB0/4", "hResidualGlobalRPhiDT_MB0/4", 200, -10, 10);
    hResidualGlobalPhiDT_MB[11] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB0/4", "hResidualGlobalPhiDT_MB0/4", 200, -1, 1);
    hResidualGlobalThetaDT_MB[11] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB0/4", "hResidualGlobalThetaDT_MB0/4", 200, -1, 1);
    hResidualGlobalZDT_MB[11] = fs->make<TH1F>("hResidualGlobalZDT_MB0/4", "hResidualGlobalZDT_MB0/4", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[12] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB1/1", "hResidualGlobalRPhiDT_MB1/1", 200, -10, 10);
    hResidualGlobalPhiDT_MB[12] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB1/1", "hResidualGlobalPhiDT_MB1/1", 200, -1, 1);
    hResidualGlobalThetaDT_MB[12] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB1/1", "hResidualGlobalThetaDT_MB1/1", 200, -1, 1);
    hResidualGlobalZDT_MB[12] = fs->make<TH1F>("hResidualGlobalZDT_MB1/1", "hResidualGlobalZDT_MB1/1", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[13] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB1/2", "hResidualGlobalRPhiDT_MB1/2", 200, -10, 10);
    hResidualGlobalPhiDT_MB[13] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB1/2", "hResidualGlobalPhiDT_MB1/2", 200, -1, 1);
    hResidualGlobalThetaDT_MB[13] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB1/2", "hResidualGlobalThetaDT_MB1/2", 200, -1, 1);
    hResidualGlobalZDT_MB[13] = fs->make<TH1F>("hResidualGlobalZDT_MB1/2", "hResidualGlobalZDT_MB1/2", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[14] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB1/3", "hResidualGlobalRPhiDT_MB1/3", 200, -10, 10);
    hResidualGlobalPhiDT_MB[14] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB1/3", "hResidualGlobalPhiDT_MB1/3", 200, -1, 1);
    hResidualGlobalThetaDT_MB[14] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB1/3", "hResidualGlobalThetaDT_MB1/3", 200, -1, 1);
    hResidualGlobalZDT_MB[14] = fs->make<TH1F>("hResidualGlobalZDT_MB1/3", "hResidualGlobalZDT_MB1/3", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[15] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB1/4", "hResidualGlobalRPhiDT_MB1/4", 200, -10, 10);
    hResidualGlobalPhiDT_MB[15] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB1/4", "hResidualGlobalPhiDT_MB1/4", 200, -1, 1);
    hResidualGlobalThetaDT_MB[15] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB1/4", "hResidualGlobalThetaDT_MB1/4", 200, -1, 1);
    hResidualGlobalZDT_MB[15] = fs->make<TH1F>("hResidualGlobalZDT_MB1/4", "hResidualGlobalZDT_MB1/4", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[16] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB2/1", "hResidualGlobalRPhiDT_MB2/1", 200, -10, 10);
    hResidualGlobalPhiDT_MB[16] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB2/1", "hResidualGlobalPhiDT_MB2/1", 200, -1, 1);
    hResidualGlobalThetaDT_MB[16] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB2/1", "hResidualGlobalThetaDT_MB2/1", 200, -1, 1);
    hResidualGlobalZDT_MB[16] = fs->make<TH1F>("hResidualGlobalZDT_MB2/1", "hResidualGlobalZDT_MB2/1", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[17] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB2/2", "hResidualGlobalRPhiDT_MB2/2", 200, -10, 10);
    hResidualGlobalPhiDT_MB[17] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB2/2", "hResidualGlobalPhiDT_MB2/2", 200, -1, 1);
    hResidualGlobalThetaDT_MB[17] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB2/2", "hResidualGlobalThetaDT_MB2/2", 200, -1, 1);
    hResidualGlobalZDT_MB[17] = fs->make<TH1F>("hResidualGlobalZDT_MB2/2", "hResidualGlobalZDT_MB2/2", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[18] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB2/3", "hResidualGlobalRPhiDT_MB2/3", 200, -10, 10);
    hResidualGlobalPhiDT_MB[18] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB2/3", "hResidualGlobalPhiDT_MB2/3", 200, -1, 1);
    hResidualGlobalThetaDT_MB[18] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB2/3", "hResidualGlobalThetaDT_MB2/3", 200, -1, 1);
    hResidualGlobalZDT_MB[18] = fs->make<TH1F>("hResidualGlobalZDT_MB2/3", "hResidualGlobalZDT_MB2/3", 200, -10, 10);
    hResidualGlobalRPhiDT_MB[19] =
        fs->make<TH1F>("hResidualGlobalRPhiDT_MB2/4", "hResidualGlobalRPhiDT_MB2/4", 200, -10, 10);
    hResidualGlobalPhiDT_MB[19] =
        fs->make<TH1F>("hResidualGlobalPhiDT_MB2/4", "hResidualGlobalPhiDT_MB2/4", 200, -1, 1);
    hResidualGlobalThetaDT_MB[19] =
        fs->make<TH1F>("hResidualGlobalThetaDT_MB2/4", "hResidualGlobalThetaDT_MB2/4", 200, -1, 1);
    hResidualGlobalZDT_MB[19] = fs->make<TH1F>("hResidualGlobalZDT_MB2/4", "hResidualGlobalZDT_MB2/4", 200, -10, 10);

    // CSC Stations
    hResidualLocalXCSC_ME[0] = fs->make<TH1F>("hResidualLocalXCSC_ME-4/1", "hResidualLocalXCSC_ME-4/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[0] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-4/1", "hResidualLocalPhiCSC_ME-4/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[0] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-4/1", "hResidualLocalThetaCSC_ME-4/1", 200, -1, 1);
    hResidualLocalYCSC_ME[0] = fs->make<TH1F>("hResidualLocalYCSC_ME-4/1", "hResidualLocalYCSC_ME-4/1", 200, -10, 10);
    hResidualLocalXCSC_ME[1] = fs->make<TH1F>("hResidualLocalXCSC_ME-4/2", "hResidualLocalXCSC_ME-4/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[1] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-4/2", "hResidualLocalPhiCSC_ME-4/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[1] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-4/2", "hResidualLocalThetaCSC_ME-4/2", 200, -1, 1);
    hResidualLocalYCSC_ME[1] = fs->make<TH1F>("hResidualLocalYCSC_ME-4/2", "hResidualLocalYCSC_ME-4/2", 200, -10, 10);
    hResidualLocalXCSC_ME[2] = fs->make<TH1F>("hResidualLocalXCSC_ME-3/1", "hResidualLocalXCSC_ME-3/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[2] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-3/1", "hResidualLocalPhiCSC_ME-3/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[2] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-3/1", "hResidualLocalThetaCSC_ME-3/1", 200, -1, 1);
    hResidualLocalYCSC_ME[2] = fs->make<TH1F>("hResidualLocalYCSC_ME-3/1", "hResidualLocalYCSC_ME-3/1", 200, -10, 10);
    hResidualLocalXCSC_ME[3] = fs->make<TH1F>("hResidualLocalXCSC_ME-3/2", "hResidualLocalXCSC_ME-3/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[3] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-3/2", "hResidualLocalPhiCSC_ME-3/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[3] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-3/2", "hResidualLocalThetaCSC_ME-3/2", 200, -1, 1);
    hResidualLocalYCSC_ME[3] = fs->make<TH1F>("hResidualLocalYCSC_ME-3/2", "hResidualLocalYCSC_ME-3/2", 200, -10, 10);
    hResidualLocalXCSC_ME[4] = fs->make<TH1F>("hResidualLocalXCSC_ME-2/1", "hResidualLocalXCSC_ME-2/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[4] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-2/1", "hResidualLocalPhiCSC_ME-2/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[4] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-2/1", "hResidualLocalThetaCSC_ME-2/1", 200, -1, 1);
    hResidualLocalYCSC_ME[4] = fs->make<TH1F>("hResidualLocalYCSC_ME-2/1", "hResidualLocalYCSC_ME-2/1", 200, -10, 10);
    hResidualLocalXCSC_ME[5] = fs->make<TH1F>("hResidualLocalXCSC_ME-2/2", "hResidualLocalXCSC_ME-2/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[5] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-2/2", "hResidualLocalPhiCSC_ME-2/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[5] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-2/2", "hResidualLocalThetaCSC_ME-2/2", 200, -1, 1);
    hResidualLocalYCSC_ME[5] = fs->make<TH1F>("hResidualLocalYCSC_ME-2/2", "hResidualLocalYCSC_ME-2/2", 200, -10, 10);
    hResidualLocalXCSC_ME[6] = fs->make<TH1F>("hResidualLocalXCSC_ME-1/1", "hResidualLocalXCSC_ME-1/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[6] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-1/1", "hResidualLocalPhiCSC_ME-1/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[6] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-1/1", "hResidualLocalThetaCSC_ME-1/1", 200, -1, 1);
    hResidualLocalYCSC_ME[6] = fs->make<TH1F>("hResidualLocalYCSC_ME-1/1", "hResidualLocalYCSC_ME-1/1", 200, -10, 10);
    hResidualLocalXCSC_ME[7] = fs->make<TH1F>("hResidualLocalXCSC_ME-1/2", "hResidualLocalXCSC_ME-1/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[7] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-1/2", "hResidualLocalPhiCSC_ME-1/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[7] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-1/2", "hResidualLocalThetaCSC_ME-1/2", 200, -1, 1);
    hResidualLocalYCSC_ME[7] = fs->make<TH1F>("hResidualLocalYCSC_ME-1/2", "hResidualLocalYCSC_ME-1/2", 200, -10, 10);
    hResidualLocalXCSC_ME[8] = fs->make<TH1F>("hResidualLocalXCSC_ME-1/3", "hResidualLocalXCSC_ME-1/3", 200, -10, 10);
    hResidualLocalPhiCSC_ME[8] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME-1/3", "hResidualLocalPhiCSC_ME-1/3", 200, -1, 1);
    hResidualLocalThetaCSC_ME[8] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME-1/3", "hResidualLocalThetaCSC_ME-1/3", 200, -1, 1);
    hResidualLocalYCSC_ME[8] = fs->make<TH1F>("hResidualLocalYCSC_ME-1/3", "hResidualLocalYCSC_ME-1/3", 200, -10, 10);
    hResidualLocalXCSC_ME[9] = fs->make<TH1F>("hResidualLocalXCSC_ME1/1", "hResidualLocalXCSC_ME1/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[9] = fs->make<TH1F>("hResidualLocalPhiCSC_ME1/1", "hResidualLocalPhiCSC_ME1/1", 100, -1, 1);
    hResidualLocalThetaCSC_ME[9] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME1/1", "hResidualLocalThetaCSC_ME1/1", 200, -1, 1);
    hResidualLocalYCSC_ME[9] = fs->make<TH1F>("hResidualLocalYCSC_ME1/1", "hResidualLocalYCSC_ME1/1", 200, -10, 10);
    hResidualLocalXCSC_ME[10] = fs->make<TH1F>("hResidualLocalXCSC_ME1/2", "hResidualLocalXCSC_ME1/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[10] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME1/2", "hResidualLocalPhiCSC_ME1/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[10] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME1/2", "hResidualLocalThetaCSC_ME1/2", 200, -1, 1);
    hResidualLocalYCSC_ME[10] = fs->make<TH1F>("hResidualLocalYCSC_ME1/2", "hResidualLocalYCSC_ME1/2", 200, -10, 10);
    hResidualLocalXCSC_ME[11] = fs->make<TH1F>("hResidualLocalXCSC_ME1/3", "hResidualLocalXCSC_ME1/3", 200, -10, 10);
    hResidualLocalPhiCSC_ME[11] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME1/3", "hResidualLocalPhiCSC_ME1/3", 200, -1, 1);
    hResidualLocalThetaCSC_ME[11] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME1/3", "hResidualLocalThetaCSC_ME1/3", 200, -1, 1);
    hResidualLocalYCSC_ME[11] = fs->make<TH1F>("hResidualLocalYCSC_ME1/3", "hResidualLocalYCSC_ME1/3", 200, -10, 10);
    hResidualLocalXCSC_ME[12] = fs->make<TH1F>("hResidualLocalXCSC_ME2/1", "hResidualLocalXCSC_ME2/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[12] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME2/1", "hResidualLocalPhiCSC_ME2/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[12] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME2/1", "hResidualLocalThetaCSC_ME2/1", 200, -1, 1);
    hResidualLocalYCSC_ME[12] = fs->make<TH1F>("hResidualLocalYCSC_ME2/1", "hResidualLocalYCSC_ME2/1", 200, -10, 10);
    hResidualLocalXCSC_ME[13] = fs->make<TH1F>("hResidualLocalXCSC_ME2/2", "hResidualLocalXCSC_ME2/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[13] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME2/2", "hResidualLocalPhiCSC_ME2/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[13] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME2/2", "hResidualLocalThetaCSC_ME2/2", 200, -1, 1);
    hResidualLocalYCSC_ME[13] = fs->make<TH1F>("hResidualLocalYCSC_ME2/2", "hResidualLocalYCSC_ME2/2", 200, -10, 10);
    hResidualLocalXCSC_ME[14] = fs->make<TH1F>("hResidualLocalXCSC_ME3/1", "hResidualLocalXCSC_ME3/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[14] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME3/1", "hResidualLocalPhiCSC_ME3/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[14] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME3/1", "hResidualLocalThetaCSC_ME3/1", 200, -1, 1);
    hResidualLocalYCSC_ME[14] = fs->make<TH1F>("hResidualLocalYCSC_ME3/1", "hResidualLocalYCSC_ME3/1", 200, -10, 10);
    hResidualLocalXCSC_ME[15] = fs->make<TH1F>("hResidualLocalXCSC_ME3/2", "hResidualLocalXCSC_ME3/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[15] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME3/2", "hResidualLocalPhiCSC_ME3/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[15] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME3/2", "hResidualLocalThetaCSC_ME3/2", 200, -1, 1);
    hResidualLocalYCSC_ME[15] = fs->make<TH1F>("hResidualLocalYCSC_ME3/2", "hResidualLocalYCSC_ME3/2", 200, -10, 10);
    hResidualLocalXCSC_ME[16] = fs->make<TH1F>("hResidualLocalXCSC_ME4/1", "hResidualLocalXCSC_ME4/1", 200, -10, 10);
    hResidualLocalPhiCSC_ME[16] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME4/1", "hResidualLocalPhiCSC_ME4/1", 200, -1, 1);
    hResidualLocalThetaCSC_ME[16] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME4/1", "hResidualLocalThetaCSC_ME4/1", 200, -1, 1);
    hResidualLocalYCSC_ME[16] = fs->make<TH1F>("hResidualLocalYCSC_ME4/1", "hResidualLocalYCSC_ME4/1", 200, -10, 10);
    hResidualLocalXCSC_ME[17] = fs->make<TH1F>("hResidualLocalXCSC_ME4/2", "hResidualLocalXCSC_ME4/2", 200, -10, 10);
    hResidualLocalPhiCSC_ME[17] =
        fs->make<TH1F>("hResidualLocalPhiCSC_ME4/2", "hResidualLocalPhiCSC_ME4/2", 200, -1, 1);
    hResidualLocalThetaCSC_ME[17] =
        fs->make<TH1F>("hResidualLocalThetaCSC_ME4/2", "hResidualLocalThetaCSC_ME4/2", 200, -1, 1);
    hResidualLocalYCSC_ME[17] = fs->make<TH1F>("hResidualLocalYCSC_ME4/2", "hResidualLocalYCSC_ME4/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[0] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-4/1", "hResidualGlobalRPhiCSC_ME-4/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[0] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-4/1", "hResidualGlobalPhiCSC_ME-4/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[0] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-4/1", "hResidualGlobalThetaCSC_ME-4/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[0] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-4/1", "hResidualGlobalRCSC_ME-4/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[1] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-4/2", "hResidualGlobalRPhiCSC_ME-4/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[1] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-4/2", "hResidualGlobalPhiCSC_ME-4/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[1] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-4/2", "hResidualGlobalThetaCSC_ME-4/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[1] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-4/2", "hResidualGlobalRCSC_ME-4/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[2] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-3/1", "hResidualGlobalRPhiCSC_ME-3/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[2] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-3/1", "hResidualGlobalPhiCSC_ME-3/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[2] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-3/1", "hResidualGlobalThetaCSC_ME-3/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[2] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-3/1", "hResidualGlobalRCSC_ME-3/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[3] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-3/2", "hResidualGlobalRPhiCSC_ME-3/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[3] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-3/2", "hResidualGlobalPhiCSC_ME-3/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[3] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-3/2", "hResidualGlobalThetaCSC_ME-3/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[3] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-3/2", "hResidualGlobalRCSC_ME-3/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[4] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-2/1", "hResidualGlobalRPhiCSC_ME-2/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[4] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-2/1", "hResidualGlobalPhiCSC_ME-2/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[4] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-2/1", "hResidualGlobalThetaCSC_ME-2/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[4] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-2/1", "hResidualGlobalRCSC_ME-2/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[5] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-2/2", "hResidualGlobalRPhiCSC_ME-2/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[5] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-2/2", "hResidualGlobalPhiCSC_ME-2/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[5] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-2/2", "hResidualGlobalThetaCSC_ME-2/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[5] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-2/2", "hResidualGlobalRCSC_ME-2/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[6] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-1/1", "hResidualGlobalRPhiCSC_ME-1/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[6] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-1/1", "hResidualGlobalPhiCSC_ME-1/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[6] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-1/1", "hResidualGlobalThetaCSC_ME-1/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[6] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-1/1", "hResidualGlobalRCSC_ME-1/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[7] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-1/2", "hResidualGlobalRPhiCSC_ME-1/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[7] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-1/2", "hResidualGlobalPhiCSC_ME-1/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[7] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-1/2", "hResidualGlobalThetaCSC_ME-1/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[7] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-1/2", "hResidualGlobalRCSC_ME-1/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[8] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME-1/3", "hResidualGlobalRPhiCSC_ME-1/3", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[8] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME-1/3", "hResidualGlobalPhiCSC_ME-1/3", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[8] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME-1/3", "hResidualGlobalThetaCSC_ME-1/3", 200, -1, 1);
    hResidualGlobalRCSC_ME[8] =
        fs->make<TH1F>("hResidualGlobalRCSC_ME-1/3", "hResidualGlobalRCSC_ME-1/3", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[9] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME1/1", "hResidualGlobalRPhiCSC_ME1/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[9] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME1/1", "hResidualGlobalPhiCSC_ME1/1", 100, -1, 1);
    hResidualGlobalThetaCSC_ME[9] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME1/1", "hResidualGlobalThetaCSC_ME1/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[9] = fs->make<TH1F>("hResidualGlobalRCSC_ME1/1", "hResidualGlobalRCSC_ME1/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[10] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME1/2", "hResidualGlobalRPhiCSC_ME1/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[10] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME1/2", "hResidualGlobalPhiCSC_ME1/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[10] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME1/2", "hResidualGlobalThetaCSC_ME1/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[10] = fs->make<TH1F>("hResidualGlobalRCSC_ME1/2", "hResidualGlobalRCSC_ME1/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[11] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME1/3", "hResidualGlobalRPhiCSC_ME1/3", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[11] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME1/3", "hResidualGlobalPhiCSC_ME1/3", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[11] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME1/3", "hResidualGlobalThetaCSC_ME1/3", 200, -1, 1);
    hResidualGlobalRCSC_ME[11] = fs->make<TH1F>("hResidualGlobalRCSC_ME1/3", "hResidualGlobalRCSC_ME1/3", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[12] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME2/1", "hResidualGlobalRPhiCSC_ME2/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[12] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME2/1", "hResidualGlobalPhiCSC_ME2/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[12] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME2/1", "hResidualGlobalThetaCSC_ME2/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[12] = fs->make<TH1F>("hResidualGlobalRCSC_ME2/1", "hResidualGlobalRCSC_ME2/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[13] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME2/2", "hResidualGlobalRPhiCSC_ME2/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[13] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME2/2", "hResidualGlobalPhiCSC_ME2/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[13] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME2/2", "hResidualGlobalThetaCSC_ME2/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[13] = fs->make<TH1F>("hResidualGlobalRCSC_ME2/2", "hResidualGlobalRCSC_ME2/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[14] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME3/1", "hResidualGlobalRPhiCSC_ME3/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[14] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME3/1", "hResidualGlobalPhiCSC_ME3/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[14] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME3/1", "hResidualGlobalThetaCSC_ME3/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[14] = fs->make<TH1F>("hResidualGlobalRCSC_ME3/1", "hResidualGlobalRCSC_ME3/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[15] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME3/2", "hResidualGlobalRPhiCSC_ME3/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[15] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME3/2", "hResidualGlobalPhiCSC_ME3/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[15] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME3/2", "hResidualGlobalThetaCSC_ME3/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[15] = fs->make<TH1F>("hResidualGlobalRCSC_ME3/2", "hResidualGlobalRCSC_ME3/2", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[16] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME4/1", "hResidualGlobalRPhiCSC_ME4/1", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[16] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME4/1", "hResidualGlobalPhiCSC_ME4/1", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[16] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME4/1", "hResidualGlobalThetaCSC_ME4/1", 200, -1, 1);
    hResidualGlobalRCSC_ME[16] = fs->make<TH1F>("hResidualGlobalRCSC_ME4/1", "hResidualGlobalRCSC_ME4/1", 200, -10, 10);
    hResidualGlobalRPhiCSC_ME[17] =
        fs->make<TH1F>("hResidualGlobalRPhiCSC_ME4/2", "hResidualGlobalRPhiCSC_ME4/2", 200, -10, 10);
    hResidualGlobalPhiCSC_ME[17] =
        fs->make<TH1F>("hResidualGlobalPhiCSC_ME4/2", "hResidualGlobalPhiCSC_ME4/2", 200, -1, 1);
    hResidualGlobalThetaCSC_ME[17] =
        fs->make<TH1F>("hResidualGlobalThetaCSC_ME4/2", "hResidualGlobalThetaCSC_ME4/2", 200, -1, 1);
    hResidualGlobalRCSC_ME[17] = fs->make<TH1F>("hResidualGlobalRCSC_ME4/2", "hResidualGlobalRCSC_ME4/2", 200, -10, 10);

    //DQM plots: mean residual with RMS as error
    hprofLocalXDT = fs->make<TH1F>("hprofLocalXDT", "Local X DT;;X (cm)", 280, 0, 280);
    hprofLocalPhiDT = fs->make<TH1F>("hprofLocalPhiDT", "Local Phi DT;;Phi (rad)", 280, 0, 280);
    hprofLocalThetaDT = fs->make<TH1F>("hprofLocalThetaDT", "Local Theta DT;;Theta (rad)", 280, 0, 280);
    hprofLocalYDT = fs->make<TH1F>("hprofLocalYDT", "Local Y DT;;Y (cm)", 280, 0, 280);
    hprofLocalXCSC = fs->make<TH1F>("hprofLocalXCSC", "Local X CSC;;X (cm)", 540, 0, 540);
    hprofLocalPhiCSC = fs->make<TH1F>("hprofLocalPhiCSC", "Local Phi CSC;;Phi (rad)", 540, 0, 540);
    hprofLocalThetaCSC = fs->make<TH1F>("hprofLocalThetaCSC", "Local Theta CSC;;Theta (rad)", 540, 0, 540);
    hprofLocalYCSC = fs->make<TH1F>("hprofLocalYCSC", "Local Y CSC;;Y (cm)", 540, 0, 540);
    hprofGlobalRPhiDT = fs->make<TH1F>("hprofGlobalRPhiDT", "Global RPhi DT;;RPhi (cm)", 280, 0, 280);
    hprofGlobalPhiDT = fs->make<TH1F>("hprofGlobalPhiDT", "Global Phi DT;;Phi (rad)", 280, 0, 280);
    hprofGlobalThetaDT = fs->make<TH1F>("hprofGlobalThetaDT", "Global Theta DT;;Theta (rad)", 280, 0, 280);
    hprofGlobalZDT = fs->make<TH1F>("hprofGlobalZDT", "Global Z DT;;Z (cm)", 280, 0, 280);
    hprofGlobalRPhiCSC = fs->make<TH1F>("hprofGlobalRPhiCSC", "Global RPhi CSC;;RPhi (cm)", 540, 0, 540);
    hprofGlobalPhiCSC = fs->make<TH1F>("hprofGlobalPhiCSC", "Global Phi CSC;;Phi (cm)", 540, 0, 540);
    hprofGlobalThetaCSC = fs->make<TH1F>("hprofGlobalThetaCSC", "Global Theta CSC;;Theta (rad)", 540, 0, 540);
    hprofGlobalRCSC = fs->make<TH1F>("hprofGlobalRCSC", "Global R CSC;;R (cm)", 540, 0, 540);

    // TH1F options
    hprofLocalXDT->GetXaxis()->SetLabelSize(0.025);
    hprofLocalPhiDT->GetXaxis()->SetLabelSize(0.025);
    hprofLocalThetaDT->GetXaxis()->SetLabelSize(0.025);
    hprofLocalYDT->GetXaxis()->SetLabelSize(0.025);
    hprofLocalXCSC->GetXaxis()->SetLabelSize(0.025);
    hprofLocalPhiCSC->GetXaxis()->SetLabelSize(0.025);
    hprofLocalThetaCSC->GetXaxis()->SetLabelSize(0.025);
    hprofLocalYCSC->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalRPhiDT->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalPhiDT->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalThetaDT->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalZDT->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalRPhiCSC->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalPhiCSC->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalThetaCSC->GetXaxis()->SetLabelSize(0.025);
    hprofGlobalRCSC->GetXaxis()->SetLabelSize(0.025);

    // TH2F histos definition
    hprofGlobalPositionDT = fs->make<TH2F>(
        "hprofGlobalPositionDT", "Global DT position (cm) absolute MEAN residuals;Sector;;cm", 14, 0, 14, 40, 0, 40);
    hprofGlobalAngleDT = fs->make<TH2F>(
        "hprofGlobalAngleDT", "Global DT angle (rad) absolute MEAN residuals;Sector;;rad", 14, 0, 14, 40, 0, 40);
    hprofGlobalPositionRmsDT = fs->make<TH2F>(
        "hprofGlobalPositionRmsDT", "Global DT position (cm) RMS residuals;Sector;;rad", 14, 0, 14, 40, 0, 40);
    hprofGlobalAngleRmsDT = fs->make<TH2F>(
        "hprofGlobalAngleRmsDT", "Global DT angle (rad) RMS residuals;Sector;;rad", 14, 0, 14, 40, 0, 40);
    hprofLocalPositionDT = fs->make<TH2F>(
        "hprofLocalPositionDT", "Local DT position (cm) absolute MEAN residuals;Sector;;cm", 14, 0, 14, 40, 0, 40);
    hprofLocalAngleDT = fs->make<TH2F>(
        "hprofLocalAngleDT", "Local DT angle (rad) absolute MEAN residuals;Sector;;rad", 14, 0, 14, 40, 0, 40);
    hprofLocalPositionRmsDT = fs->make<TH2F>(
        "hprofLocalPositionRmsDT", "Local DT position (cm) RMS residuals;Sector;;rad", 14, 0, 14, 40, 0, 40);
    hprofLocalAngleRmsDT =
        fs->make<TH2F>("hprofLocalAngleRmsDT", "Local DT angle (rad) RMS residuals;Sector;;rad", 14, 0, 14, 40, 0, 40);

    hprofGlobalPositionCSC = fs->make<TH2F>(
        "hprofGlobalPositionCSC", "Global CSC position (cm) absolute MEAN residuals;Sector;;cm", 36, 0, 36, 36, 0, 36);
    hprofGlobalAngleCSC = fs->make<TH2F>(
        "hprofGlobalAngleCSC", "Global CSC angle (rad) absolute MEAN residuals;Sector;;rad", 36, 0, 36, 36, 0, 36);
    hprofGlobalPositionRmsCSC = fs->make<TH2F>(
        "hprofGlobalPositionRmsCSC", "Global CSC position (cm) RMS residuals;Sector;;rad", 36, 0, 36, 36, 0, 36);
    hprofGlobalAngleRmsCSC = fs->make<TH2F>(
        "hprofGlobalAngleRmsCSC", "Global CSC angle (rad) RMS residuals;Sector;;rad", 36, 0, 36, 36, 0, 36);
    hprofLocalPositionCSC = fs->make<TH2F>(
        "hprofLocalPositionCSC", "Local CSC position (cm) absolute MEAN residuals;Sector;;cm", 36, 0, 36, 36, 0, 36);
    hprofLocalAngleCSC = fs->make<TH2F>(
        "hprofLocalAngleCSC", "Local CSC angle (rad) absolute MEAN residuals;Sector;;rad", 36, 0, 36, 36, 0, 36);
    hprofLocalPositionRmsCSC = fs->make<TH2F>(
        "hprofLocalPositionRmsCSC", "Local CSC position (cm) RMS residuals;Sector;;rad", 36, 0, 36, 36, 0, 36);
    hprofLocalAngleRmsCSC = fs->make<TH2F>(
        "hprofLocalAngleRmsCSC", "Local CSC angle (rad) RMS residuals;Sector;;rad", 36, 0, 36, 36, 0, 36);

    // histos options
    Float_t labelSize = 0.025;
    hprofGlobalPositionDT->GetYaxis()->SetLabelSize(labelSize);
    hprofGlobalAngleDT->GetYaxis()->SetLabelSize(labelSize);
    hprofGlobalPositionRmsDT->GetYaxis()->SetLabelSize(labelSize);
    hprofGlobalAngleRmsDT->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalPositionDT->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalAngleDT->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalPositionRmsDT->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalAngleRmsDT->GetYaxis()->SetLabelSize(labelSize);

    hprofGlobalPositionCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofGlobalAngleCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofGlobalPositionRmsCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofGlobalAngleRmsCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalPositionCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalAngleCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalPositionRmsCSC->GetYaxis()->SetLabelSize(labelSize);
    hprofLocalAngleRmsCSC->GetYaxis()->SetLabelSize(labelSize);

    char binLabel[32];
    for (int i = 1; i < 15; i++) {
      snprintf(binLabel, sizeof(binLabel), "Sec-%d", i);
      hprofGlobalPositionDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofGlobalAngleDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofGlobalPositionRmsDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofGlobalAngleRmsDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalPositionDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalAngleDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalPositionRmsDT->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalAngleRmsDT->GetXaxis()->SetBinLabel(i, binLabel);
    }

    for (int i = 1; i < 37; i++) {
      snprintf(binLabel, sizeof(binLabel), "Ch-%d", i);
      hprofGlobalPositionCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofGlobalAngleCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofGlobalPositionRmsCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofGlobalAngleRmsCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalPositionCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalAngleCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalPositionRmsCSC->GetXaxis()->SetBinLabel(i, binLabel);
      hprofLocalAngleRmsCSC->GetXaxis()->SetBinLabel(i, binLabel);
    }
  }
}

void MuonAlignmentAnalyzer::endJob() {
  edm::LogInfo("MuonAlignmentAnalyzer") << "----------------- " << std::endl << std::endl;

  if (theDataType == "SimData")
    edm::LogInfo("MuonAlignmentAnalyzer") << "Number of Sim tracks: " << numberOfSimTracks << std::endl << std::endl;

  if (doSAplots)
    edm::LogInfo("MuonAlignmentAnalyzer") << "Number of SA Reco tracks: " << numberOfSARecTracks << std::endl
                                          << std::endl;

  if (doGBplots)
    edm::LogInfo("MuonAlignmentAnalyzer") << "Number of GB Reco tracks: " << numberOfGBRecTracks << std::endl
                                          << std::endl;

  if (doResplots) {
    //  delete thePropagator;

    edm::LogInfo("MuonAlignmentAnalyzer") << "Number of Hits considered for residuals: " << numberOfHits << std::endl
                                          << std::endl;

    char binLabel[40];

    for (unsigned int i = 0; i < unitsLocalX.size(); i++) {
      TString nameHistoLocalX = unitsLocalX[i]->GetName();

      TString nameHistoLocalPhi = unitsLocalPhi[i]->GetName();

      TString nameHistoLocalTheta = unitsLocalTheta[i]->GetName();

      TString nameHistoLocalY = unitsLocalY[i]->GetName();

      TString nameHistoGlobalRPhi = unitsGlobalRPhi[i]->GetName();

      TString nameHistoGlobalPhi = unitsGlobalPhi[i]->GetName();

      TString nameHistoGlobalTheta = unitsGlobalTheta[i]->GetName();

      TString nameHistoGlobalRZ = unitsGlobalRZ[i]->GetName();

      if (nameHistoLocalX.Contains("MB"))  // HistoLocalX DT
      {
        int wheel, station, sector;

        sscanf(nameHistoLocalX, "ResidualLocalX_W%dMB%1dS%d", &wheel, &station, &sector);

        Int_t nstation = station - 1;
        Int_t nwheel = wheel + 2;

        Double_t MeanRPhi = unitsLocalX[i]->GetMean();
        Double_t ErrorRPhi = unitsLocalX[i]->GetMeanError();

        Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

        snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

        hprofLocalXDT->SetMarkerStyle(21);
        hprofLocalXDT->SetMarkerColor(kRed);
        hprofLocalXDT->SetBinContent(xbin, MeanRPhi);
        hprofLocalXDT->SetBinError(xbin, ErrorRPhi);
        hprofLocalXDT->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = 1 + nwheel * 8 + nstation * 2;
        hprofLocalPositionDT->SetBinContent(sector, ybin, fabs(MeanRPhi));
        snprintf(binLabel, sizeof(binLabel), "MB%d/%d_LocalX", wheel, station);
        hprofLocalPositionDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofLocalPositionRmsDT->SetBinContent(sector, ybin, ErrorRPhi);
        hprofLocalPositionRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoLocalX.Contains("ME"))  // HistoLocalX CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoLocalX, "ResidualLocalX_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanRPhi = unitsLocalX[i]->GetMean();
        Double_t ErrorRPhi = unitsLocalX[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofLocalXCSC->SetMarkerStyle(21);
        hprofLocalXCSC->SetMarkerColor(kRed);
        hprofLocalXCSC->SetBinContent(xbin, MeanRPhi);
        hprofLocalXCSC->SetBinError(xbin, ErrorRPhi);
        hprofLocalXCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin - 1;
        hprofLocalPositionCSC->SetBinContent(chamber, ybin, fabs(MeanRPhi));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_LocalX", station, ring);
        hprofLocalPositionCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofLocalPositionRmsCSC->SetBinContent(chamber, ybin, ErrorRPhi);
        hprofLocalPositionRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoLocalTheta.Contains("MB"))  // HistoLocalTheta DT
      {
        int wheel, station, sector;

        sscanf(nameHistoLocalTheta, "ResidualLocalTheta_W%dMB%1dS%d", &wheel, &station, &sector);

        if (station != 4) {
          Int_t nstation = station - 1;
          Int_t nwheel = wheel + 2;

          Double_t MeanTheta = unitsLocalTheta[i]->GetMean();
          Double_t ErrorTheta = unitsLocalTheta[i]->GetMeanError();

          Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

          snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

          hprofLocalThetaDT->SetBinContent(xbin, MeanTheta);
          hprofLocalThetaDT->SetBinError(xbin, ErrorTheta);
          hprofLocalThetaDT->SetMarkerStyle(21);
          hprofLocalThetaDT->SetMarkerColor(kRed);
          hprofLocalThetaDT->GetXaxis()->SetBinLabel(xbin, binLabel);

          Int_t ybin = 2 + nwheel * 8 + nstation * 2;
          hprofLocalAngleDT->SetBinContent(sector, ybin, fabs(MeanTheta));
          snprintf(binLabel, sizeof(binLabel), "MB%d/%d_LocalTheta", wheel, station);
          hprofLocalAngleDT->GetYaxis()->SetBinLabel(ybin, binLabel);
          hprofLocalAngleRmsDT->SetBinContent(sector, ybin, ErrorTheta);
          hprofLocalAngleRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        }
      }

      if (nameHistoLocalPhi.Contains("MB"))  // HistoLocalPhi DT
      {
        int wheel, station, sector;

        sscanf(nameHistoLocalPhi, "ResidualLocalPhi_W%dMB%1dS%d", &wheel, &station, &sector);

        Int_t nstation = station - 1;
        Int_t nwheel = wheel + 2;

        Double_t MeanPhi = unitsLocalPhi[i]->GetMean();
        Double_t ErrorPhi = unitsLocalPhi[i]->GetMeanError();

        Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

        snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

        hprofLocalPhiDT->SetBinContent(xbin, MeanPhi);
        hprofLocalPhiDT->SetBinError(xbin, ErrorPhi);
        hprofLocalPhiDT->SetMarkerStyle(21);
        hprofLocalPhiDT->SetMarkerColor(kRed);
        hprofLocalPhiDT->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = 1 + nwheel * 8 + nstation * 2;
        hprofLocalAngleDT->SetBinContent(sector, ybin, fabs(MeanPhi));
        snprintf(binLabel, sizeof(binLabel), "MB%d/%d_LocalPhi", wheel, station);
        hprofLocalAngleDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofLocalAngleRmsDT->SetBinContent(sector, ybin, ErrorPhi);
        hprofLocalAngleRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoLocalPhi.Contains("ME"))  // HistoLocalPhi CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoLocalPhi, "ResidualLocalPhi_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanPhi = unitsLocalPhi[i]->GetMean();
        Double_t ErrorPhi = unitsLocalPhi[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofLocalPhiCSC->SetMarkerStyle(21);
        hprofLocalPhiCSC->SetMarkerColor(kRed);
        hprofLocalPhiCSC->SetBinContent(xbin, MeanPhi);
        hprofLocalPhiCSC->SetBinError(xbin, ErrorPhi);
        hprofLocalPhiCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin - 1;
        hprofLocalAngleCSC->SetBinContent(chamber, ybin, fabs(MeanPhi));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_LocalPhi", station, ring);
        hprofLocalAngleCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofLocalAngleRmsCSC->SetBinContent(chamber, ybin, ErrorPhi);
        hprofLocalAngleRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoLocalTheta.Contains("ME"))  // HistoLocalTheta CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoLocalTheta, "ResidualLocalTheta_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanTheta = unitsLocalTheta[i]->GetMean();
        Double_t ErrorTheta = unitsLocalTheta[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofLocalThetaCSC->SetMarkerStyle(21);
        hprofLocalThetaCSC->SetMarkerColor(kRed);
        hprofLocalThetaCSC->SetBinContent(xbin, MeanTheta);
        hprofLocalThetaCSC->SetBinError(xbin, ErrorTheta);
        hprofLocalThetaCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin;
        hprofLocalAngleCSC->SetBinContent(chamber, ybin, fabs(MeanTheta));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_LocalTheta", station, ring);
        hprofLocalAngleCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofLocalAngleRmsCSC->SetBinContent(chamber, ybin, ErrorTheta);
        hprofLocalAngleRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoLocalY.Contains("MB"))  // HistoLocalY DT
      {
        int wheel, station, sector;

        sscanf(nameHistoLocalY, "ResidualLocalY_W%dMB%1dS%d", &wheel, &station, &sector);

        if (station != 4) {
          Int_t nstation = station - 1;
          Int_t nwheel = wheel + 2;

          Double_t MeanZ = unitsLocalY[i]->GetMean();
          Double_t ErrorZ = unitsLocalY[i]->GetMeanError();

          Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

          snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

          hprofLocalYDT->SetMarkerStyle(21);
          hprofLocalYDT->SetMarkerColor(kRed);
          hprofLocalYDT->SetBinContent(xbin, MeanZ);
          hprofLocalYDT->SetBinError(xbin, ErrorZ);
          hprofLocalYDT->GetXaxis()->SetBinLabel(xbin, binLabel);

          Int_t ybin = 2 + nwheel * 8 + nstation * 2;
          hprofLocalPositionDT->SetBinContent(sector, ybin, fabs(MeanZ));
          snprintf(binLabel, sizeof(binLabel), "MB%d/%d_LocalY", wheel, station);
          hprofLocalPositionDT->GetYaxis()->SetBinLabel(ybin, binLabel);
          hprofLocalPositionRmsDT->SetBinContent(sector, ybin, ErrorZ);
          hprofLocalPositionRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        }
      }

      if (nameHistoLocalY.Contains("ME"))  // HistoLocalY CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoLocalY, "ResidualLocalY_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanR = unitsLocalY[i]->GetMean();
        Double_t ErrorR = unitsLocalY[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofLocalYCSC->SetMarkerStyle(21);
        hprofLocalYCSC->SetMarkerColor(kRed);
        hprofLocalYCSC->SetBinContent(xbin, MeanR);
        hprofLocalYCSC->SetBinError(xbin, ErrorR);
        hprofLocalYCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin;
        hprofLocalPositionCSC->SetBinContent(chamber, ybin, fabs(MeanR));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_LocalY", station, ring);
        hprofLocalPositionCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofLocalPositionRmsCSC->SetBinContent(chamber, ybin, ErrorR);
        hprofLocalPositionRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoGlobalRPhi.Contains("MB"))  // HistoGlobalRPhi DT
      {
        int wheel, station, sector;

        sscanf(nameHistoGlobalRPhi, "ResidualGlobalRPhi_W%dMB%1dS%d", &wheel, &station, &sector);

        Int_t nstation = station - 1;
        Int_t nwheel = wheel + 2;

        Double_t MeanRPhi = unitsGlobalRPhi[i]->GetMean();
        Double_t ErrorRPhi = unitsGlobalRPhi[i]->GetMeanError();

        Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

        snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

        hprofGlobalRPhiDT->SetMarkerStyle(21);
        hprofGlobalRPhiDT->SetMarkerColor(kRed);
        hprofGlobalRPhiDT->SetBinContent(xbin, MeanRPhi);
        hprofGlobalRPhiDT->SetBinError(xbin, ErrorRPhi);
        hprofGlobalRPhiDT->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = 1 + nwheel * 8 + nstation * 2;
        hprofGlobalPositionDT->SetBinContent(sector, ybin, fabs(MeanRPhi));
        snprintf(binLabel, sizeof(binLabel), "MB%d/%d_GlobalRPhi", wheel, station);
        hprofGlobalPositionDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofGlobalPositionRmsDT->SetBinContent(sector, ybin, ErrorRPhi);
        hprofGlobalPositionRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoGlobalRPhi.Contains("ME"))  // HistoGlobalRPhi CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoGlobalRPhi, "ResidualGlobalRPhi_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanRPhi = unitsGlobalRPhi[i]->GetMean();
        Double_t ErrorRPhi = unitsGlobalRPhi[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofGlobalRPhiCSC->SetMarkerStyle(21);
        hprofGlobalRPhiCSC->SetMarkerColor(kRed);
        hprofGlobalRPhiCSC->SetBinContent(xbin, MeanRPhi);
        hprofGlobalRPhiCSC->SetBinError(xbin, ErrorRPhi);
        hprofGlobalRPhiCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin - 1;
        hprofGlobalPositionCSC->SetBinContent(chamber, ybin, fabs(MeanRPhi));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_GlobalRPhi", station, ring);
        hprofGlobalPositionCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofGlobalPositionRmsCSC->SetBinContent(chamber, ybin, ErrorRPhi);
        hprofGlobalPositionRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoGlobalTheta.Contains("MB"))  // HistoGlobalRTheta DT
      {
        int wheel, station, sector;

        sscanf(nameHistoGlobalTheta, "ResidualGlobalTheta_W%dMB%1dS%d", &wheel, &station, &sector);

        if (station != 4) {
          Int_t nstation = station - 1;
          Int_t nwheel = wheel + 2;

          Double_t MeanTheta = unitsGlobalTheta[i]->GetMean();
          Double_t ErrorTheta = unitsGlobalTheta[i]->GetMeanError();

          Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

          snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

          hprofGlobalThetaDT->SetBinContent(xbin, MeanTheta);
          hprofGlobalThetaDT->SetBinError(xbin, ErrorTheta);
          hprofGlobalThetaDT->SetMarkerStyle(21);
          hprofGlobalThetaDT->SetMarkerColor(kRed);
          hprofGlobalThetaDT->GetXaxis()->SetBinLabel(xbin, binLabel);

          Int_t ybin = 2 + nwheel * 8 + nstation * 2;
          hprofGlobalAngleDT->SetBinContent(sector, ybin, fabs(MeanTheta));
          snprintf(binLabel, sizeof(binLabel), "MB%d/%d_GlobalTheta", wheel, station);
          hprofGlobalAngleDT->GetYaxis()->SetBinLabel(ybin, binLabel);
          hprofGlobalAngleRmsDT->SetBinContent(sector, ybin, ErrorTheta);
          hprofGlobalAngleRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        }
      }

      if (nameHistoGlobalPhi.Contains("MB"))  // HistoGlobalPhi DT
      {
        int wheel, station, sector;

        sscanf(nameHistoGlobalPhi, "ResidualGlobalPhi_W%dMB%1dS%d", &wheel, &station, &sector);

        Int_t nstation = station - 1;
        Int_t nwheel = wheel + 2;

        Double_t MeanPhi = unitsGlobalPhi[i]->GetMean();
        Double_t ErrorPhi = unitsGlobalPhi[i]->GetMeanError();

        Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

        snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

        hprofGlobalPhiDT->SetBinContent(xbin, MeanPhi);
        hprofGlobalPhiDT->SetBinError(xbin, ErrorPhi);
        hprofGlobalPhiDT->SetMarkerStyle(21);
        hprofGlobalPhiDT->SetMarkerColor(kRed);
        hprofGlobalPhiDT->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = 1 + nwheel * 8 + nstation * 2;
        hprofGlobalAngleDT->SetBinContent(sector, ybin, fabs(MeanPhi));
        snprintf(binLabel, sizeof(binLabel), "MB%d/%d_GlobalPhi", wheel, station);
        hprofGlobalAngleDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofGlobalAngleRmsDT->SetBinContent(sector, ybin, ErrorPhi);
        hprofGlobalAngleRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoGlobalPhi.Contains("ME"))  // HistoGlobalPhi CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoGlobalPhi, "ResidualGlobalPhi_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanPhi = unitsGlobalPhi[i]->GetMean();
        Double_t ErrorPhi = unitsGlobalPhi[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofGlobalPhiCSC->SetMarkerStyle(21);
        hprofGlobalPhiCSC->SetMarkerColor(kRed);
        hprofGlobalPhiCSC->SetBinContent(xbin, MeanPhi);
        hprofGlobalPhiCSC->SetBinError(xbin, ErrorPhi);
        hprofGlobalPhiCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin - 1;
        hprofGlobalAngleCSC->SetBinContent(chamber, ybin, fabs(MeanPhi));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_GlobalPhi", station, ring);
        hprofGlobalAngleCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofGlobalAngleRmsCSC->SetBinContent(chamber, ybin, ErrorPhi);
        hprofGlobalAngleRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoGlobalTheta.Contains("ME"))  // HistoGlobalTheta CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoGlobalTheta, "ResidualGlobalTheta_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanTheta = unitsGlobalTheta[i]->GetMean();
        Double_t ErrorTheta = unitsGlobalTheta[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofGlobalThetaCSC->SetMarkerStyle(21);
        hprofGlobalThetaCSC->SetMarkerColor(kRed);
        hprofGlobalThetaCSC->SetBinContent(xbin, MeanTheta);
        hprofGlobalThetaCSC->SetBinError(xbin, ErrorTheta);
        hprofGlobalThetaCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin;
        hprofGlobalAngleCSC->SetBinContent(chamber, ybin, fabs(MeanTheta));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_GlobalTheta", station, ring);
        hprofGlobalAngleCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofGlobalAngleRmsCSC->SetBinContent(chamber, ybin, ErrorTheta);
        hprofGlobalAngleRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

      if (nameHistoGlobalRZ.Contains("MB"))  // HistoGlobalZ DT
      {
        int wheel, station, sector;

        sscanf(nameHistoGlobalRZ, "ResidualGlobalZ_W%dMB%1dS%d", &wheel, &station, &sector);

        if (station != 4) {
          Int_t nstation = station - 1;
          Int_t nwheel = wheel + 2;

          Double_t MeanZ = unitsGlobalRZ[i]->GetMean();
          Double_t ErrorZ = unitsGlobalRZ[i]->GetMeanError();

          Int_t xbin = sector + 14 * nstation + 14 * 4 * nwheel;

          snprintf(binLabel, sizeof(binLabel), "MB%d/%dS%d", wheel, station, sector);

          hprofGlobalZDT->SetMarkerStyle(21);
          hprofGlobalZDT->SetMarkerColor(kRed);

          hprofGlobalZDT->SetBinContent(xbin, MeanZ);
          hprofGlobalZDT->SetBinError(xbin, ErrorZ);
          hprofGlobalZDT->GetXaxis()->SetBinLabel(xbin, binLabel);

          Int_t ybin = 2 + nwheel * 8 + nstation * 2;
          hprofGlobalPositionDT->SetBinContent(sector, ybin, fabs(MeanZ));
          snprintf(binLabel, sizeof(binLabel), "MB%d/%d_GlobalZ", wheel, station);
          hprofGlobalPositionDT->GetYaxis()->SetBinLabel(ybin, binLabel);
          hprofGlobalPositionRmsDT->SetBinContent(sector, ybin, ErrorZ);
          hprofGlobalPositionRmsDT->GetYaxis()->SetBinLabel(ybin, binLabel);
        }
      }

      if (nameHistoGlobalRZ.Contains("ME"))  // HistoGlobalR CSC
      {
        int station, ring, chamber;

        sscanf(nameHistoGlobalRZ, "ResidualGlobalR_ME%dR%1dC%d", &station, &ring, &chamber);

        Double_t MeanR = unitsGlobalRZ[i]->GetMean();
        Double_t ErrorR = unitsGlobalRZ[i]->GetMeanError();

        Int_t xbin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          xbin = ring;
        if (station > 0)
          xbin = xbin + 9;
        else
          xbin = 10 - xbin;

        // To avoid holes in xAxis, I can't imagine right now a simpler way...
        if (xbin < 5)
          xbin = 18 * (((Int_t)(xbin / 3)) * 2 + (Int_t)(xbin / 2)) + chamber;
        else if (xbin < 6)
          xbin = 108 + chamber;
        else if (xbin < 14)
          xbin = 126 + (xbin - 6) * 36 + chamber;
        else if (xbin < 18)
          xbin = 414 + 18 * (((Int_t)(xbin - 13) / 3) * 2 + ((Int_t)(xbin - 13) / 2)) + chamber;
        else
          xbin = 522 + chamber;

        snprintf(binLabel, sizeof(binLabel), "ME%d/%dC%d", station, ring, chamber);

        hprofGlobalRCSC->SetMarkerStyle(21);
        hprofGlobalRCSC->SetMarkerColor(kRed);
        hprofGlobalRCSC->SetBinContent(xbin, MeanR);
        hprofGlobalRCSC->SetBinError(xbin, ErrorR);
        hprofGlobalRCSC->GetXaxis()->SetBinLabel(xbin, binLabel);

        Int_t ybin = abs(station) * 2 + ring;
        if (abs(station) == 1)
          ybin = ring;
        if (station > 0)
          ybin = ybin + 9;
        else
          ybin = 10 - ybin;
        ybin = 2 * ybin;
        hprofGlobalPositionCSC->SetBinContent(chamber, ybin, fabs(MeanR));
        snprintf(binLabel, sizeof(binLabel), "ME%d/%d_GlobalR", station, ring);
        hprofGlobalPositionCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
        hprofGlobalPositionRmsCSC->SetBinContent(chamber, ybin, ErrorR);
        hprofGlobalPositionRmsCSC->GetYaxis()->SetBinLabel(ybin, binLabel);
      }

    }  // for in histos

  }  // doResPlots
}

void MuonAlignmentAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &eventSetup) {
  GlobalVector p1, p2;
  std::vector<double> simPar[4];  //pt,eta,phi,charge

  // ######### if data= MC, do Simulation Plots#####
  if (theDataType == "SimData") {
    double simEta = 0;
    double simPt = 0;
    double simPhi = 0;
    int i = 0, ie = 0, ib = 0;

    // Get the SimTrack collection from the event
    const edm::Handle<edm::SimTrackContainer> &simTracks = event.getHandle(simTrackToken_);

    edm::SimTrackContainer::const_iterator simTrack;

    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack) {
      if (abs((*simTrack).type()) == 13) {
        i++;
        simPt = (*simTrack).momentum().Pt();
        simEta = (*simTrack).momentum().eta();
        simPhi = (*simTrack).momentum().phi();
        numberOfSimTracks++;
        hSimPT->Fill(simPt);
        if (fabs(simEta) < 1.04) {
          hSimPT_Barrel->Fill(simPt);
          ib++;
        } else {
          hSimPT_Endcap->Fill(simPt);
          ie++;
        }
        hSimPTvsEta->Fill(simEta, simPt);
        hSimPTvsPhi->Fill(simPhi, simPt);
        hSimPhivsEta->Fill(simEta, simPhi);

        simPar[0].push_back(simPt);
        simPar[1].push_back(simEta);
        simPar[2].push_back(simPhi);
        simPar[3].push_back((*simTrack).charge());

        //	Save the muon pair
        if (i == 1)
          p1 = GlobalVector((*simTrack).momentum().x(), (*simTrack).momentum().y(), (*simTrack).momentum().z());
        if (i == 2)
          p2 = GlobalVector((*simTrack).momentum().x(), (*simTrack).momentum().y(), (*simTrack).momentum().z());
      }
    }
    hSimNmuons->Fill(i);
    hSimNmuons_Barrel->Fill(ib);
    hSimNmuons_Endcap->Fill(ie);

    if (i > 1) {  // Take 2 first muons :-(
      TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
      TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
      TLorentzVector pair = mu1 + mu2;
      double Minv = pair.M();
      hSimInvM->Fill(Minv);
      if (fabs(p1.eta()) < 1.04 && fabs(p2.eta()) < 1.04)
        hSimInvM_Barrel->Fill(Minv);
      else if (fabs(p1.eta()) >= 1.04 && fabs(p2.eta()) >= 1.04)
        hSimInvM_Endcap->Fill(Minv);
      else
        hSimInvM_Overlap->Fill(Minv);
    }

  }  //simData

  // ############ Stand Alone Muon plots ###############
  if (doSAplots) {
    double SArecPt = 0.;
    double SAeta = 0.;
    double SAphi = 0.;
    int i = 0, ie = 0, ib = 0;
    double ich = 0;

    // Get the RecTrack collection from the event
    const edm::Handle<reco::TrackCollection> &staTracks = event.getHandle(staTrackToken_);
    numberOfSARecTracks += staTracks->size();

    reco::TrackCollection::const_iterator staTrack;

    for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {
      i++;

      SArecPt = (*staTrack).pt();
      SAeta = (*staTrack).eta();
      SAphi = (*staTrack).phi();
      ich = (*staTrack).charge();

      hSAPTRec->Fill(SArecPt);
      hSAPhivsEta->Fill(SAeta, SAphi);
      hSAChi2->Fill((*staTrack).chi2());
      hSANhits->Fill((*staTrack).numberOfValidHits());
      if (fabs(SAeta) < 1.04) {
        hSAPTRec_Barrel->Fill(SArecPt);
        hSAChi2_Barrel->Fill((*staTrack).chi2());
        hSANhits_Barrel->Fill((*staTrack).numberOfValidHits());
        ib++;
      } else {
        hSAPTRec_Endcap->Fill(SArecPt);
        hSAChi2_Endcap->Fill((*staTrack).chi2());
        hSANhits_Endcap->Fill((*staTrack).numberOfValidHits());
        ie++;
      }

      // save the muon pair
      if (i == 1)
        p1 = GlobalVector((*staTrack).momentum().x(), (*staTrack).momentum().y(), (*staTrack).momentum().z());
      if (i == 2)
        p2 = GlobalVector((*staTrack).momentum().x(), (*staTrack).momentum().y(), (*staTrack).momentum().z());

      if (SArecPt && theDataType == "SimData") {
        double candDeltaR = -999.0, dR;
        int iCand = 0;
        if (!simPar[0].empty()) {
          for (unsigned int iSim = 0; iSim < simPar[0].size(); iSim++) {
            dR = deltaR(SAeta, SAphi, simPar[1][iSim], simPar[2][iSim]);
            if (candDeltaR < 0 || dR < candDeltaR) {
              candDeltaR = dR;
              iCand = iSim;
            }
          }
        }

        double simPt = simPar[0][iCand];
        hSAPTres->Fill((SArecPt - simPt) / simPt);
        if (fabs(SAeta) < 1.04)
          hSAPTres_Barrel->Fill((SArecPt - simPt) / simPt);
        else
          hSAPTres_Endcap->Fill((SArecPt - simPt) / simPt);

        hSAPTDiff->Fill(SArecPt - simPt);

        hSAPTDiffvsEta->Fill(SAeta, SArecPt - simPt);
        hSAPTDiffvsPhi->Fill(SAphi, SArecPt - simPt);
        double ptInvRes = (ich / SArecPt - simPar[3][iCand] / simPt) / (simPar[3][iCand] / simPt);
        hSAinvPTres->Fill(ptInvRes);

        hSAinvPTvsEta->Fill(SAeta, ptInvRes);
        hSAinvPTvsPhi->Fill(SAphi, ptInvRes);
        hSAinvPTvsNhits->Fill((*staTrack).numberOfValidHits(), ptInvRes);
      }

      hSAPTvsEta->Fill(SAeta, SArecPt);
      hSAPTvsPhi->Fill(SAphi, SArecPt);
    }

    hSANmuons->Fill(i);
    hSANmuons_Barrel->Fill(ib);
    hSANmuons_Endcap->Fill(ie);

    if (i > 1) {  // Take 2 first muons :-(
      TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
      TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
      TLorentzVector pair = mu1 + mu2;
      double Minv = pair.M();
      hSAInvM->Fill(Minv);
      if (fabs(p1.eta()) < 1.04 && fabs(p2.eta()) < 1.04)
        hSAInvM_Barrel->Fill(Minv);
      else if (fabs(p1.eta()) >= 1.04 && fabs(p2.eta()) >= 1.04)
        hSAInvM_Endcap->Fill(Minv);
      else
        hSAInvM_Overlap->Fill(Minv);
    }  // 2 first muons

  }  //end doSAplots

  // ############### Global Muons plots ##########

  if (doGBplots) {
    // Get the RecTrack collection from the event
    const edm::Handle<reco::TrackCollection> &glbTracks = event.getHandle(glbTrackToken_);
    numberOfGBRecTracks += glbTracks->size();

    double GBrecPt = 0;
    double GBeta = 0;
    double GBphi = 0;
    double ich = 0;
    int i = 0, ie = 0, ib = 0;

    reco::TrackCollection::const_iterator glbTrack;

    for (glbTrack = glbTracks->begin(); glbTrack != glbTracks->end(); ++glbTrack) {
      i++;

      GBrecPt = (*glbTrack).pt();
      GBeta = (*glbTrack).eta();
      GBphi = (*glbTrack).phi();
      ich = (*glbTrack).charge();

      hGBPTRec->Fill(GBrecPt);
      hGBPhivsEta->Fill(GBeta, GBphi);
      hGBChi2->Fill((*glbTrack).chi2());
      hGBNhits->Fill((*glbTrack).numberOfValidHits());
      if (fabs(GBeta) < 1.04) {
        hGBPTRec_Barrel->Fill(GBrecPt);
        hGBChi2_Barrel->Fill((*glbTrack).chi2());
        hGBNhits_Barrel->Fill((*glbTrack).numberOfValidHits());
        ib++;
      } else {
        hGBPTRec_Endcap->Fill(GBrecPt);
        hGBChi2_Endcap->Fill((*glbTrack).chi2());
        hGBNhits_Endcap->Fill((*glbTrack).numberOfValidHits());
        ie++;
      }

      // save the muon pair
      if (i == 1)
        p1 = GlobalVector((*glbTrack).momentum().x(), (*glbTrack).momentum().y(), (*glbTrack).momentum().z());
      if (i == 2)
        p2 = GlobalVector((*glbTrack).momentum().x(), (*glbTrack).momentum().y(), (*glbTrack).momentum().z());

      if (GBrecPt && theDataType == "SimData") {
        double candDeltaR = -999.0, dR;
        int iCand = 0;
        if (!simPar[0].empty()) {
          for (unsigned int iSim = 0; iSim < simPar[0].size(); iSim++) {
            dR = deltaR(GBeta, GBphi, simPar[1][iSim], simPar[2][iSim]);
            if (candDeltaR < 0 || dR < candDeltaR) {
              candDeltaR = dR;
              iCand = iSim;
            }
          }
        }

        double simPt = simPar[0][iCand];

        hGBPTres->Fill((GBrecPt - simPt) / simPt);
        if (fabs(GBeta) < 1.04)
          hGBPTres_Barrel->Fill((GBrecPt - simPt) / simPt);
        else
          hGBPTres_Endcap->Fill((GBrecPt - simPt) / simPt);

        hGBPTDiff->Fill(GBrecPt - simPt);

        hGBPTDiffvsEta->Fill(GBeta, GBrecPt - simPt);
        hGBPTDiffvsPhi->Fill(GBphi, GBrecPt - simPt);

        double ptInvRes = (ich / GBrecPt - simPar[3][iCand] / simPt) / (simPar[3][iCand] / simPt);
        hGBinvPTres->Fill(ptInvRes);

        hGBinvPTvsEta->Fill(GBeta, ptInvRes);
        hGBinvPTvsPhi->Fill(GBphi, ptInvRes);
        hGBinvPTvsNhits->Fill((*glbTrack).numberOfValidHits(), ptInvRes);
      }

      hGBPTvsEta->Fill(GBeta, GBrecPt);
      hGBPTvsPhi->Fill(GBphi, GBrecPt);
    }

    hGBNmuons->Fill(i);
    hGBNmuons_Barrel->Fill(ib);
    hGBNmuons_Endcap->Fill(ie);

    if (i > 1) {  // Take 2 first muons :-(
      TLorentzVector mu1(p1.x(), p1.y(), p1.z(), p1.mag());
      TLorentzVector mu2(p2.x(), p2.y(), p2.z(), p2.mag());
      TLorentzVector pair = mu1 + mu2;
      double Minv = pair.M();
      hGBInvM->Fill(Minv);
      if (fabs(p1.eta()) < 1.04 && fabs(p2.eta()) < 1.04)
        hGBInvM_Barrel->Fill(Minv);
      else if (fabs(p1.eta()) >= 1.04 && fabs(p2.eta()) >= 1.04)
        hGBInvM_Endcap->Fill(Minv);
      else
        hGBInvM_Overlap->Fill(Minv);
    }

  }  //end doGBplots

  // ############    Residual plots ###################

  if (doResplots) {
    const MagneticField *theMGField = &eventSetup.getData(magFieldToken_);
    const edm::ESHandle<GlobalTrackingGeometry> &theTrackingGeometry = eventSetup.getHandle(trackingGeometryToken_);

    // Get the RecTrack collection from the event
    const edm::Handle<reco::TrackCollection> &staTracks = event.getHandle(staTrackToken_);

    // Get the 4D DTSegments
    const edm::Handle<DTRecSegment4DCollection> &all4DSegmentsDT = event.getHandle(allDTSegmentToken_);
    DTRecSegment4DCollection::const_iterator segmentDT;

    // Get the 4D CSCSegments
    const edm::Handle<CSCSegmentCollection> &all4DSegmentsCSC = event.getHandle(allCSCSegmentToken_);
    CSCSegmentCollection::const_iterator segmentCSC;

    //Vectors used to perform the matching between Segments and hits from Track
    intDVector indexCollectionDT;
    intDVector indexCollectionCSC;

    /*    std::cout << "<MuonAlignmentAnalyzer> List of DTSegments found in Local Reconstruction" << std::endl;
      std::cout << "Number: " << all4DSegmentsDT->size() << std::endl;
      for (segmentDT = all4DSegmentsDT->begin(); segmentDT != all4DSegmentsDT->end(); ++segmentDT){
      const GeomDet* geomDet = theTrackingGeometry->idToDet((*segmentDT).geographicalId());
      std::cout << "<MuonAlignmentAnalyzer> " << geomDet->toGlobal((*segmentDT).localPosition()) << std::endl;
      std::cout << "<MuonAlignmentAnalyzer> Local " << (*segmentDT).localPosition() << std::endl;
      }
      std::cout << "<MuonAlignmentAnalyzer> List of CSCSegments found in Local Reconstruction" << std::endl;
      for (segmentCSC = all4DSegmentsCSC->begin(); segmentCSC != all4DSegmentsCSC->end(); ++segmentCSC){
      const GeomDet* geomDet = theTrackingGeometry->idToDet((*segmentCSC).geographicalId());
      std::cout << "<MuonAlignmentAnalyzer>" << geomDet->toGlobal((*segmentCSC).localPosition()) << std::endl;
      }
*/
    thePropagator = new SteppingHelixPropagator(theMGField, alongMomentum);

    reco::TrackCollection::const_iterator staTrack;
    for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {
      int countPoints = 0;

      reco::TransientTrack track(*staTrack, theMGField, theTrackingGeometry);

      if (staTrack->numberOfValidHits() > (min1DTrackRecHitSize - 1)) {
        RecHitVector my4DTrack = this->doMatching(
            *staTrack, all4DSegmentsDT, all4DSegmentsCSC, &indexCollectionDT, &indexCollectionCSC, theTrackingGeometry);

        //cut in number of segments

        if (my4DTrack.size() > (min4DTrackSegmentSize - 1)) {
          // start propagation
          //    TrajectoryStateOnSurface innerTSOS = track.impactPointState();
          TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();

          //If the state is valid
          if (innerTSOS.isValid()) {
            //Loop over Associated segments
            for (RecHitVector::iterator rechit = my4DTrack.begin(); rechit != my4DTrack.end(); ++rechit) {
              const GeomDet *geomDet = theTrackingGeometry->idToDet((*rechit)->geographicalId());
              //Otherwise the propagator could throw an exception
              const Plane *pDest = dynamic_cast<const Plane *>(&geomDet->surface());
              const Cylinder *cDest = dynamic_cast<const Cylinder *>(&geomDet->surface());

              if (pDest != nullptr || cDest != nullptr) {  //Donde antes iba el try

                TrajectoryStateOnSurface destiny =
                    thePropagator->propagate(*(innerTSOS.freeState()), geomDet->surface());

                if (!destiny.isValid() || !destiny.hasError())
                  continue;

                /*	    std::cout << "<MuonAlignmentAnalyzer> Segment: " << geomDet->toGlobal((*rechit)->localPosition()) << std::endl;
  std::cout << "<MuonAlignmentAnalyzer> Segment local: " << (*rechit)->localPosition() << std::endl;
  std::cout << "<MuonAlignmentAnalyzer> Predicted: " << destiny.freeState()->position() << std::endl;
  std::cout << "<MuonAlignmentAnalyzer> Predicted local: " << destiny.localPosition() << std::endl;
*/
                const long rawId = (*rechit)->geographicalId().rawId();
                int position = -1;
                bool newDetector = true;
                //Loop over the DetectorCollection to see if the detector is new and requires a new entry
                for (std::vector<long>::iterator myIds = detectorCollection.begin(); myIds != detectorCollection.end();
                     myIds++) {
                  ++position;
                  //If matches newDetector = false
                  if (*myIds == rawId) {
                    newDetector = false;
                    break;
                  }
                }

                DetId myDet(rawId);
                int det = myDet.subdetId();
                int wheel = 0, station = 0, sector = 0;
                int endcap = 0, ring = 0, chamber = 0;

                double residualGlobalRPhi = 0, residualGlobalPhi = 0, residualGlobalR = 0, residualGlobalTheta = 0,
                       residualGlobalZ = 0;
                double residualLocalX = 0, residualLocalPhi = 0, residualLocalY = 0, residualLocalTheta = 0;

                // Fill generic histograms
                //If it's a DT
                if (det == 1) {
                  DTChamberId myChamber(rawId);
                  wheel = myChamber.wheel();
                  station = myChamber.station();
                  sector = myChamber.sector();

                  //global
                  residualGlobalRPhi =
                      geomDet->toGlobal((*rechit)->localPosition()).perp() *
                          geomDet->toGlobal((*rechit)->localPosition()).barePhi() -
                      destiny.freeState()->position().perp() * destiny.freeState()->position().barePhi();

                  //local
                  residualLocalX = (*rechit)->localPosition().x() - destiny.localPosition().x();

                  //global
                  residualGlobalPhi = geomDet->toGlobal(((RecSegment *)(*rechit))->localDirection()).barePhi() -
                                      destiny.globalDirection().barePhi();

                  //local
                  residualLocalPhi = atan2(((RecSegment *)(*rechit))->localDirection().z(),
                                           ((RecSegment *)(*rechit))->localDirection().x()) -
                                     atan2(destiny.localDirection().z(), destiny.localDirection().x());

                  hResidualGlobalRPhiDT->Fill(residualGlobalRPhi);
                  hResidualGlobalPhiDT->Fill(residualGlobalPhi);
                  hResidualLocalXDT->Fill(residualLocalX);
                  hResidualLocalPhiDT->Fill(residualLocalPhi);

                  if (station != 4) {
                    //global
                    residualGlobalZ =
                        geomDet->toGlobal((*rechit)->localPosition()).z() - destiny.freeState()->position().z();

                    //local
                    residualLocalY = (*rechit)->localPosition().y() - destiny.localPosition().y();

                    //global
                    residualGlobalTheta = geomDet->toGlobal(((RecSegment *)(*rechit))->localDirection()).bareTheta() -
                                          destiny.globalDirection().bareTheta();

                    //local
                    residualLocalTheta = atan2(((RecSegment *)(*rechit))->localDirection().z(),
                                               ((RecSegment *)(*rechit))->localDirection().y()) -
                                         atan2(destiny.localDirection().z(), destiny.localDirection().y());

                    hResidualGlobalThetaDT->Fill(residualGlobalTheta);
                    hResidualGlobalZDT->Fill(residualGlobalZ);
                    hResidualLocalThetaDT->Fill(residualLocalTheta);
                    hResidualLocalYDT->Fill(residualLocalY);
                  }

                  int index = wheel + 2;
                  hResidualGlobalRPhiDT_W[index]->Fill(residualGlobalRPhi);
                  hResidualGlobalPhiDT_W[index]->Fill(residualGlobalPhi);
                  hResidualLocalXDT_W[index]->Fill(residualLocalX);
                  hResidualLocalPhiDT_W[index]->Fill(residualLocalPhi);
                  if (station != 4) {
                    hResidualGlobalThetaDT_W[index]->Fill(residualGlobalTheta);
                    hResidualGlobalZDT_W[index]->Fill(residualGlobalZ);
                    hResidualLocalThetaDT_W[index]->Fill(residualLocalTheta);
                    hResidualLocalYDT_W[index]->Fill(residualLocalY);
                  }

                  index = wheel * 4 + station + 7;
                  hResidualGlobalRPhiDT_MB[index]->Fill(residualGlobalRPhi);
                  hResidualGlobalPhiDT_MB[index]->Fill(residualGlobalPhi);
                  hResidualLocalXDT_MB[index]->Fill(residualLocalX);
                  hResidualLocalPhiDT_MB[index]->Fill(residualLocalPhi);

                  if (station != 4) {
                    hResidualGlobalThetaDT_MB[index]->Fill(residualGlobalTheta);
                    hResidualGlobalZDT_MB[index]->Fill(residualGlobalZ);
                    hResidualLocalThetaDT_MB[index]->Fill(residualLocalTheta);
                    hResidualLocalYDT_MB[index]->Fill(residualLocalY);
                  }
                } else if (det == 2) {
                  CSCDetId myChamber(rawId);
                  endcap = myChamber.endcap();
                  station = myChamber.station();
                  if (endcap == 2)
                    station = -station;
                  ring = myChamber.ring();
                  chamber = myChamber.chamber();

                  //global
                  residualGlobalRPhi =
                      geomDet->toGlobal((*rechit)->localPosition()).perp() *
                          geomDet->toGlobal((*rechit)->localPosition()).barePhi() -
                      destiny.freeState()->position().perp() * destiny.freeState()->position().barePhi();

                  //local
                  residualLocalX = (*rechit)->localPosition().x() - destiny.localPosition().x();

                  //global
                  residualGlobalR =
                      geomDet->toGlobal((*rechit)->localPosition()).perp() - destiny.freeState()->position().perp();

                  //local
                  residualLocalY = (*rechit)->localPosition().y() - destiny.localPosition().y();

                  //global
                  residualGlobalPhi = geomDet->toGlobal(((RecSegment *)(*rechit))->localDirection()).barePhi() -
                                      destiny.globalDirection().barePhi();

                  //local
                  residualLocalPhi = atan2(((RecSegment *)(*rechit))->localDirection().y(),
                                           ((RecSegment *)(*rechit))->localDirection().x()) -
                                     atan2(destiny.localDirection().y(), destiny.localDirection().x());

                  //global
                  residualGlobalTheta = geomDet->toGlobal(((RecSegment *)(*rechit))->localDirection()).bareTheta() -
                                        destiny.globalDirection().bareTheta();

                  //local
                  residualLocalTheta = atan2(((RecSegment *)(*rechit))->localDirection().y(),
                                             ((RecSegment *)(*rechit))->localDirection().z()) -
                                       atan2(destiny.localDirection().y(), destiny.localDirection().z());

                  hResidualGlobalRPhiCSC->Fill(residualGlobalRPhi);
                  hResidualGlobalPhiCSC->Fill(residualGlobalPhi);
                  hResidualGlobalThetaCSC->Fill(residualGlobalTheta);
                  hResidualGlobalRCSC->Fill(residualGlobalR);
                  hResidualLocalXCSC->Fill(residualLocalX);
                  hResidualLocalPhiCSC->Fill(residualLocalPhi);
                  hResidualLocalThetaCSC->Fill(residualLocalTheta);
                  hResidualLocalYCSC->Fill(residualLocalY);

                  int index = 2 * station + ring + 7;
                  if (station == -1) {
                    index = 5 + ring;
                    if (ring == 4)
                      index = 6;
                  }
                  if (station == 1) {
                    index = 8 + ring;
                    if (ring == 4)
                      index = 9;
                  }
                  hResidualGlobalRPhiCSC_ME[index]->Fill(residualGlobalRPhi);
                  hResidualGlobalPhiCSC_ME[index]->Fill(residualGlobalPhi);
                  hResidualGlobalThetaCSC_ME[index]->Fill(residualGlobalTheta);
                  hResidualGlobalRCSC_ME[index]->Fill(residualGlobalR);
                  hResidualLocalXCSC_ME[index]->Fill(residualLocalX);
                  hResidualLocalPhiCSC_ME[index]->Fill(residualLocalPhi);
                  hResidualLocalThetaCSC_ME[index]->Fill(residualLocalTheta);
                  hResidualLocalYCSC_ME[index]->Fill(residualLocalY);

                } else {
                  residualGlobalRPhi = 0, residualGlobalPhi = 0, residualGlobalR = 0, residualGlobalTheta = 0,
                  residualGlobalZ = 0;
                  residualLocalX = 0, residualLocalPhi = 0, residualLocalY = 0, residualLocalTheta = 0;
                }
                // Fill individual chamber histograms
                if (newDetector) {
                  //Create an RawIdDetector, fill it and push it into the collection
                  detectorCollection.push_back(rawId);

                  //This piece of code calculates the range of the residuals
                  double rangeX = 3.0, rangeY = 5.;
                  switch (abs(station)) {
                    case 1: {
                      rangeX = resLocalXRangeStation1;
                      rangeY = resLocalYRangeStation1;
                    } break;
                    case 2: {
                      rangeX = resLocalXRangeStation2;
                      rangeY = resLocalYRangeStation2;
                    } break;
                    case 3: {
                      rangeX = resLocalXRangeStation3;
                      rangeY = resLocalYRangeStation3;
                    } break;
                    case 4: {
                      rangeX = resLocalXRangeStation4;
                      rangeY = resLocalYRangeStation4;
                    } break;
                    default:
                      break;
                  }

                  //create new histograms

                  char nameOfHistoLocalX[50];
                  char nameOfHistoLocalTheta[50];
                  char nameOfHistoLocalY[50];
                  char nameOfHistoLocalPhi[50];
                  char nameOfHistoGlobalRPhi[50];
                  char nameOfHistoGlobalTheta[50];
                  char nameOfHistoGlobalR[50];
                  char nameOfHistoGlobalPhi[50];
                  char nameOfHistoGlobalZ[50];

                  if (det == 1) {  // DT
                    snprintf(nameOfHistoLocalX,
                             sizeof(nameOfHistoLocalX),
                             "ResidualLocalX_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    snprintf(nameOfHistoLocalPhi,
                             sizeof(nameOfHistoLocalPhi),
                             "ResidualLocalPhi_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    snprintf(nameOfHistoGlobalRPhi,
                             sizeof(nameOfHistoGlobalRPhi),
                             "ResidualGlobalRPhi_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    snprintf(nameOfHistoGlobalPhi,
                             sizeof(nameOfHistoGlobalPhi),
                             "ResidualGlobalPhi_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    snprintf(nameOfHistoLocalTheta,
                             sizeof(nameOfHistoLocalTheta),
                             "ResidualLocalTheta_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    snprintf(nameOfHistoLocalY,
                             sizeof(nameOfHistoLocalY),
                             "ResidualLocalY_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    TH1F *histoLocalY = fs->make<TH1F>(nameOfHistoLocalY, nameOfHistoLocalY, nbins, -rangeY, rangeY);
                    unitsLocalY.push_back(histoLocalY);
                    snprintf(nameOfHistoGlobalTheta,
                             sizeof(nameOfHistoGlobalTheta),
                             "ResidualGlobalTheta_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    snprintf(nameOfHistoGlobalZ,
                             sizeof(nameOfHistoGlobalZ),
                             "ResidualGlobalZ_W%dMB%1dS%1d",
                             wheel,
                             station,
                             sector);
                    TH1F *histoGlobalZ = fs->make<TH1F>(nameOfHistoGlobalZ, nameOfHistoGlobalZ, nbins, -rangeY, rangeY);
                    unitsGlobalRZ.push_back(histoGlobalZ);

                  } else if (det == 2) {  //CSC
                    snprintf(nameOfHistoLocalX,
                             sizeof(nameOfHistoLocalX),
                             "ResidualLocalX_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    snprintf(nameOfHistoLocalPhi,
                             sizeof(nameOfHistoLocalPhi),
                             "ResidualLocalPhi_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    snprintf(nameOfHistoLocalTheta,
                             sizeof(nameOfHistoLocalTheta),
                             "ResidualLocalTheta_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    snprintf(nameOfHistoLocalY,
                             sizeof(nameOfHistoLocalY),
                             "ResidualLocalY_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    TH1F *histoLocalY = fs->make<TH1F>(nameOfHistoLocalY, nameOfHistoLocalY, nbins, -rangeY, rangeY);
                    unitsLocalY.push_back(histoLocalY);
                    snprintf(nameOfHistoGlobalRPhi,
                             sizeof(nameOfHistoGlobalRPhi),
                             "ResidualGlobalRPhi_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    snprintf(nameOfHistoGlobalPhi,
                             sizeof(nameOfHistoGlobalPhi),
                             "ResidualGlobalPhi_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    snprintf(nameOfHistoGlobalTheta,
                             sizeof(nameOfHistoGlobalTheta),
                             "ResidualGlobalTheta_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    snprintf(nameOfHistoGlobalR,
                             sizeof(nameOfHistoGlobalR),
                             "ResidualGlobalR_ME%dR%1dC%1d",
                             station,
                             ring,
                             chamber);
                    TH1F *histoGlobalR = fs->make<TH1F>(nameOfHistoGlobalR, nameOfHistoGlobalR, nbins, -rangeY, rangeY);
                    unitsGlobalRZ.push_back(histoGlobalR);
                  }

                  // Common histos to DT and CSC
                  TH1F *histoLocalX = fs->make<TH1F>(nameOfHistoLocalX, nameOfHistoLocalX, nbins, -rangeX, rangeX);
                  TH1F *histoGlobalRPhi =
                      fs->make<TH1F>(nameOfHistoGlobalRPhi, nameOfHistoGlobalRPhi, nbins, -rangeX, rangeX);
                  TH1F *histoLocalPhi =
                      fs->make<TH1F>(nameOfHistoLocalPhi, nameOfHistoLocalPhi, nbins, -resPhiRange, resPhiRange);
                  TH1F *histoGlobalPhi =
                      fs->make<TH1F>(nameOfHistoGlobalPhi, nameOfHistoGlobalPhi, nbins, -resPhiRange, resPhiRange);
                  TH1F *histoGlobalTheta = fs->make<TH1F>(
                      nameOfHistoGlobalTheta, nameOfHistoGlobalTheta, nbins, -resThetaRange, resThetaRange);
                  TH1F *histoLocalTheta = fs->make<TH1F>(
                      nameOfHistoLocalTheta, nameOfHistoLocalTheta, nbins, -resThetaRange, resThetaRange);

                  histoLocalX->Fill(residualLocalX);
                  histoLocalPhi->Fill(residualLocalPhi);
                  histoLocalTheta->Fill(residualLocalTheta);
                  histoGlobalRPhi->Fill(residualGlobalRPhi);
                  histoGlobalPhi->Fill(residualGlobalPhi);
                  histoGlobalTheta->Fill(residualGlobalTheta);
                  //Push them into their respective vectors
                  unitsLocalX.push_back(histoLocalX);
                  unitsLocalPhi.push_back(histoLocalPhi);
                  unitsLocalTheta.push_back(histoLocalTheta);
                  unitsGlobalRPhi.push_back(histoGlobalRPhi);
                  unitsGlobalPhi.push_back(histoGlobalPhi);
                  unitsGlobalTheta.push_back(histoGlobalTheta);

                }  // new detector
                else {
                  //If the detector was not new, just fill the histogram
                  unitsLocalX.at(position)->Fill(residualLocalX);
                  unitsLocalPhi.at(position)->Fill(residualLocalPhi);
                  unitsLocalTheta.at(position)->Fill(residualLocalTheta);
                  unitsGlobalRPhi.at(position)->Fill(residualGlobalRPhi);
                  unitsGlobalPhi.at(position)->Fill(residualGlobalPhi);
                  unitsGlobalTheta.at(position)->Fill(residualGlobalTheta);
                  if (det == 1) {
                    unitsLocalY.at(position)->Fill(residualLocalY);
                    unitsGlobalRZ.at(position)->Fill(residualGlobalZ);
                  } else if (det == 2) {
                    unitsLocalY.at(position)->Fill(residualLocalY);
                    unitsGlobalRZ.at(position)->Fill(residualGlobalR);
                  }
                }

                countPoints++;

                innerTSOS = destiny;

              } else {
                edm::LogError("MuonAlignmentAnalyzer") << " Error!! Exception in propagator catched" << std::endl;
                continue;
              }

            }  //loop over my4DTrack
          }    //TSOS was valid

        }  // cut in at least 4 segments

      }  //end cut in RecHitsSize>36
      numberOfHits = numberOfHits + countPoints;
    }  //loop over STAtracks

    delete thePropagator;

  }  //end doResplots
}

RecHitVector MuonAlignmentAnalyzer::doMatching(const reco::Track &staTrack,
                                               const edm::Handle<DTRecSegment4DCollection> &all4DSegmentsDT,
                                               const edm::Handle<CSCSegmentCollection> &all4DSegmentsCSC,
                                               intDVector *indexCollectionDT,
                                               intDVector *indexCollectionCSC,
                                               const edm::ESHandle<GlobalTrackingGeometry> &theTrackingGeometry) {
  DTRecSegment4DCollection::const_iterator segmentDT;
  CSCSegmentCollection::const_iterator segmentCSC;

  std::vector<int> positionDT;
  std::vector<int> positionCSC;
  RecHitVector my4DTrack;

  //Loop over the hits of the track
  for (int counter = 0; counter != staTrack.numberOfValidHits() - 1; counter++) {
    TrackingRecHitRef myRef = staTrack.recHit(counter);
    const TrackingRecHit *rechit = myRef.get();
    const GeomDet *geomDet = theTrackingGeometry->idToDet(rechit->geographicalId());

    //It's a DT Hit
    if (geomDet->subDetector() == GeomDetEnumerators::DT) {
      //Take the layer associated to this hit
      DTLayerId myLayer(rechit->geographicalId().rawId());

      int NumberOfDTSegment = 0;
      //Loop over segments
      for (segmentDT = all4DSegmentsDT->begin(); segmentDT != all4DSegmentsDT->end(); ++segmentDT) {
        //By default the chamber associated to this Segment is new
        bool isNewChamber = true;

        //Loop over segments already included in the vector of segments in the actual track
        for (std::vector<int>::iterator positionIt = positionDT.begin(); positionIt != positionDT.end(); positionIt++) {
          //If this segment has been used before isNewChamber = false
          if (NumberOfDTSegment == *positionIt)
            isNewChamber = false;
        }

        //Loop over vectors of segments associated to previous tracks
        for (std::vector<std::vector<int> >::iterator collect = indexCollectionDT->begin();
             collect != indexCollectionDT->end();
             ++collect) {
          //Loop over segments associated to a track
          for (std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end();
               positionIt++) {
            //If this segment was used in a previos track then isNewChamber = false
            if (NumberOfDTSegment == *positionIt)
              isNewChamber = false;
          }
        }

        //If the chamber is new
        if (isNewChamber) {
          DTChamberId myChamber((*segmentDT).geographicalId().rawId());
          //If the layer of the hit belongs to the chamber of the 4D Segment
          if (myLayer.wheel() == myChamber.wheel() && myLayer.station() == myChamber.station() &&
              myLayer.sector() == myChamber.sector()) {
            //push position of the segment and tracking rechit
            positionDT.push_back(NumberOfDTSegment);
            my4DTrack.push_back((TrackingRecHit *)&(*segmentDT));
          }
        }
        NumberOfDTSegment++;
      }
      //In case is a CSC
    } else if (geomDet->subDetector() == GeomDetEnumerators::CSC) {
      //Take the layer associated to this hit
      CSCDetId myLayer(rechit->geographicalId().rawId());

      int NumberOfCSCSegment = 0;
      //Loop over 4Dsegments
      for (segmentCSC = all4DSegmentsCSC->begin(); segmentCSC != all4DSegmentsCSC->end(); segmentCSC++) {
        //By default the chamber associated to the segment is new
        bool isNewChamber = true;

        //Loop over segments in the current track
        for (std::vector<int>::iterator positionIt = positionCSC.begin(); positionIt != positionCSC.end();
             positionIt++) {
          //If this segment has been used then newchamber = false
          if (NumberOfCSCSegment == *positionIt)
            isNewChamber = false;
        }
        //Loop over vectors of segments in previous tracks
        for (std::vector<std::vector<int> >::iterator collect = indexCollectionCSC->begin();
             collect != indexCollectionCSC->end();
             ++collect) {
          //Loop over segments in a track
          for (std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end();
               positionIt++) {
            //If the segment was used in a previous track isNewChamber = false
            if (NumberOfCSCSegment == *positionIt)
              isNewChamber = false;
          }
        }
        //If the chamber is new
        if (isNewChamber) {
          CSCDetId myChamber((*segmentCSC).geographicalId().rawId());
          //If the chambers are the same
          if (myLayer.chamberId() == myChamber.chamberId()) {
            //push
            positionCSC.push_back(NumberOfCSCSegment);
            my4DTrack.push_back((TrackingRecHit *)&(*segmentCSC));
          }
        }
        NumberOfCSCSegment++;
      }
    }
  }

  indexCollectionDT->push_back(positionDT);
  indexCollectionCSC->push_back(positionCSC);

  return my4DTrack;
}

DEFINE_FWK_MODULE(MuonAlignmentAnalyzer);

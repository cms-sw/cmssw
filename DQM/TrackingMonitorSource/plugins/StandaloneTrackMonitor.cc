#include "TFile.h"
#include "TH1.h"
#include "TMath.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TPRegexp.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h" 
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DQM/TrackingMonitorSource/interface/StandaloneTrackMonitor.h"
// -----------------------------
// constructors and destructor
// -----------------------------
StandaloneTrackMonitor::StandaloneTrackMonitor(const edm::ParameterSet& ps): 
  parameters_(ps),
  moduleName_(parameters_.getUntrackedParameter<std::string>("moduleName", "StandaloneTrackMonitor")),
  folderName_(parameters_.getUntrackedParameter<std::string>("folderName", "highPurityTracks")),
  trackTag_(parameters_.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
  bsTag_(parameters_.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
  vertexTag_(parameters_.getUntrackedParameter<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"))),
  //primaryVertexInputTag_(parameters_.getUntrackedParameter<edm::InputTag>("primaryVertexInputTag", edm::InputTag("primaryVertex"))),
  puSummaryTag_(parameters_.getUntrackedParameter<edm::InputTag>("puTag", edm::InputTag("addPileupInfo"))),
  clusterTag_(parameters_.getUntrackedParameter<edm::InputTag>("clusterTag", edm::InputTag("siStripClusters"))),
  trackToken_(consumes<reco::TrackCollection>(trackTag_)),
  bsToken_(consumes<reco::BeamSpot>(bsTag_)),
  vertexToken_(consumes<reco::VertexCollection>(vertexTag_)),
  //pvToken_(consumes<reco::VertexCollection>(primaryVertexInputTag_)),
  puSummaryToken_(consumes<std::vector<PileupSummaryInfo> >(puSummaryTag_)),
  clusterToken_(consumes<edmNew::DetSetVector<SiStripCluster> >(clusterTag_)),
  trackQuality_(parameters_.getUntrackedParameter<std::string>("trackQuality", "highPurity")),
  doPUCorrection_(parameters_.getUntrackedParameter<bool>("doPUCorrection", false)),
  isMC_(parameters_.getUntrackedParameter<bool>("isMC", false)),
  haveAllHistograms_(parameters_.getUntrackedParameter<bool>("haveAllHistograms", false)),
  puScaleFactorFile_(parameters_.getUntrackedParameter<std::string>("puScaleFactorFile", "PileupScaleFactor.root")),
  verbose_(parameters_.getUntrackedParameter<bool>("verbose", false))
{
  // for MC only
  nVtxH_ = nullptr;
  nVertexH_ = nullptr;
  bunchCrossingH_ = nullptr;
  nPUH_ = nullptr;
  trueNIntH_ = nullptr;

  nLostHitsVspTH_ = nullptr;
  nLostHitsVsEtaH_ = nullptr;
  nLostHitsVsCosThetaH_ = nullptr;
  nLostHitsVsPhiH_ = nullptr;

  nHitsTIBSVsEtaH_ = nullptr;
  nHitsTOBSVsEtaH_ = nullptr;
  nHitsTECSVsEtaH_ = nullptr;
  nHitsTIDSVsEtaH_ = nullptr;
  nHitsStripSVsEtaH_ = nullptr;

  nHitsTIBDVsEtaH_ = nullptr;
  nHitsTOBDVsEtaH_ = nullptr;
  nHitsTECDVsEtaH_ = nullptr;
  nHitsTIDDVsEtaH_ = nullptr;
  nHitsStripDVsEtaH_ = nullptr;

  nValidHitsVspTH_ = nullptr;
  nValidHitsVsEtaH_ = nullptr;
  nValidHitsVsCosThetaH_ = nullptr;
  nValidHitsVsPhiH_ = nullptr;
  nValidHitsVsnVtxH_ = nullptr;

  nValidHitsPixVsEtaH_ = nullptr;
  nValidHitsPixBVsEtaH_ = nullptr;
  nValidHitsPixEVsEtaH_ = nullptr;
  nValidHitsStripVsEtaH_ = nullptr;
  nValidHitsTIBVsEtaH_ = nullptr;
  nValidHitsTOBVsEtaH_ = nullptr;
  nValidHitsTECVsEtaH_ = nullptr;
  nValidHitsTIDVsEtaH_ = nullptr;

  nValidHitsPixVsPhiH_ = nullptr;
  nValidHitsPixBVsPhiH_ = nullptr;
  nValidHitsPixEVsPhiH_ = nullptr;
  nValidHitsStripVsPhiH_ = nullptr;
  nValidHitsTIBVsPhiH_ = nullptr;
  nValidHitsTOBVsPhiH_ = nullptr;
  nValidHitsTECVsPhiH_ = nullptr;
  nValidHitsTIDVsPhiH_ = nullptr;

  nLostHitsPixVsEtaH_ = nullptr;
  nLostHitsPixBVsEtaH_ = nullptr;
  nLostHitsPixEVsEtaH_ = nullptr;
  nLostHitsStripVsEtaH_ = nullptr;
  nLostHitsTIBVsEtaH_ = nullptr;
  nLostHitsTOBVsEtaH_ = nullptr;
  nLostHitsTECVsEtaH_ = nullptr;
  nLostHitsTIDVsEtaH_ = nullptr;

  nLostHitsPixVsPhiH_ = nullptr;
  nLostHitsPixBVsPhiH_ = nullptr;
  nLostHitsPixEVsPhiH_ = nullptr;
  nLostHitsStripVsPhiH_ = nullptr;
  nLostHitsTIBVsPhiH_ = nullptr;
  nLostHitsTOBVsPhiH_ = nullptr;
  nLostHitsTECVsPhiH_ = nullptr;
  nLostHitsTIDVsPhiH_ = nullptr;

  hOnTrkClusChargeThinH_ = nullptr;
  hOnTrkClusWidthThinH_ = nullptr;
  hOnTrkClusChargeThickH_ = nullptr;
  hOnTrkClusWidthThickH_ = nullptr;
  
  hOffTrkClusChargeThinH_ = nullptr;
  hOffTrkClusWidthThinH_ = nullptr;
  hOffTrkClusChargeThickH_ = nullptr;
  hOffTrkClusWidthThickH_ = nullptr;

  // Read pileup weight factors
  if (isMC_ && doPUCorrection_) {
    vpu_.clear();
    TFile* f1 = TFile::Open(puScaleFactorFile_.c_str());
    TH1F* h1 = dynamic_cast<TH1F*>(f1->Get("pileupweight"));
    for (int i = 1; i <= h1->GetNbinsX(); ++i) vpu_.push_back(h1->GetBinContent(i));
    f1->Close();
  }
}

void StandaloneTrackMonitor::bookHistograms(DQMStore::IBooker &iBook, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  
  edm::ParameterSet TrackEtaHistoPar = parameters_.getParameter<edm::ParameterSet>("trackEtaH");
  edm::ParameterSet TrackPtHistoPar = parameters_.getParameter<edm::ParameterSet>("trackPtH");

  std::string currentFolder = moduleName_ + "/" + folderName_ ;
  iBook.setCurrentFolder(currentFolder.c_str());

  // The following are common with the official tool
  if (haveAllHistograms_) {
    trackEtaH_ = iBook.book1D("trackEta", "Track Eta", 
			      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
			      TrackEtaHistoPar.getParameter<double>("Xmin"), 
			      TrackEtaHistoPar.getParameter<double>("Xmax"));
    
    trackEtaerrH_ = iBook.book1D("trackEtaerr", "Track Eta Error", 50,0.0,1.0);
    trackCosThetaH_ = iBook.book1D("trackCosTheta", "Track Cos(Theta)", 50,-1.0,1.0);
    trackThetaerrH_ = iBook.book1D("trackThetaerr", "Track Theta Error", 50,0.0,1.0);
    trackPhiH_ = iBook.book1D("trackPhi", "Track Phi", 70,-3.5,3.5);
    trackPhierrH_ = iBook.book1D("trackPhierr", "Track Phi Error", 50,0.0,1.0);
    
    trackPH_ = iBook.book1D("trackP", "Track 4-momentum", 50,0.0,10.0);
    trackPtH_ = iBook.book1D("trackPt", "Track Pt", 
			     TrackPtHistoPar.getParameter<int32_t>("Xbins"),
			     TrackPtHistoPar.getParameter<double>("Xmin"),
			     TrackPtHistoPar.getParameter<double>("Xmax"));
    trackPtUpto2GeVH_ = iBook.book1D("trackPtUpto2GeV", "Track Pt upto 2GeV",100,0,2.0);
    trackPtOver10GeVH_ = iBook.book1D("trackPtOver10GeV","Track Pt greater than 10 GeV",100,0,100.0);
    trackPterrH_ = iBook.book1D("trackPterr", "Track Pt Error",100,0.0,100.0);
    trackqOverpH_ = iBook.book1D("trackqOverp", "q Over p",40,-10.0,10.0);
    trackqOverperrH_ = iBook.book1D("trackqOverperr","q Over p Error",50,0.0,25.0);
    trackChargeH_ = iBook.book1D("trackCharge", "Track Charge", 50, -5, 5);
    trackChi2H_ = iBook.book1D("trackChi2", "Chi2",100,0.0,100.0);
    tracknDOFH_ = iBook.book1D("tracknDOF", "nDOF",100,0.0,100.0);
    trackChi2ProbH_ = iBook.book1D("trackChi2Prob", "Chi2prob",50,0.0,1.0);
    trackChi2oNDFH_ = iBook.book1D("trackChi2oNDF", "Chi2oNDF",100,0.0,100.0);
    trackd0H_ = iBook.book1D("trackd0", "Track d0",100,-1,1);
    trackChi2bynDOFH_ = iBook.book1D("trackChi2bynDOF", "Chi2 Over nDOF",100,0.0,10.0);
    
    DistanceOfClosestApproachToPVH_ = iBook.book1D("DistanceOfClosestApproachToPV", "DistanceOfClosestApproachToPV",100,-0.5,0.5);
    DistanceOfClosestApproachToPVVsPhiH_ = iBook.bookProfile("DistanceOfClosestApproachToPVVsPhi", "DistanceOfClosestApproachToPVVsPhi",100,-3.5,3.5,0.0,0.0,"g");
    xPointOfClosestApproachVsZ0wrtPVH_ = iBook.bookProfile("xPointOfClosestApproachVsZ0wrtPV", "xPointOfClosestApproachVsZ0wrtPV",120,-60,60,0.0,0.0,"g");
    yPointOfClosestApproachVsZ0wrtPVH_ = iBook.bookProfile("yPointOfClosestApproachVsZ0wrtPV", "yPointOfClosestApproachVsZ0wrtPV",120,-60,60,0.0,0.0,"g");
    
    sip3dToPVH_ = iBook.book1D("sip3dToPV", "signed Impact Point 3d To PV",200,-10,10);
    sip2dToPVH_ = iBook.book1D("sip2dToPV", "signed Impact Point 2d To PV",200,-10,10);
    sipDxyToPVH_ = iBook.book1D("sipDxyToPV", "signed Impact Point dxy To PV",100,-10,10);
    sipDzToPVH_ = iBook.book1D("sipDzToPV", "signed Impact Point dz To PV",100,-10,10);
    
    nvalidTrackerHitsH_ = iBook.book1D("nvalidTrackerhits", "No. of Valid Tracker Hits",45,0.5,45.5);
    nvalidPixelHitsH_ = iBook.book1D("nvalidPixelHits", "No. of Valid Hits in Pixel",6,-0.5,5.5);
    nvalidPixelBHitsH_ = iBook.book1D("nvalidPixelBarrelHits", "No. of Valid Hits in Pixel Barrel",5,-0.5,4.5);
    nvalidPixelEHitsH_ = iBook.book1D("nvalidPixelEndcapHits", "No. of Valid Hits in Pixel Endcap",5,-0.5,5.5);
    nvalidStripHitsH_ = iBook.book1D("nvalidStripHits", "No. of Valid Hits in Strip",35,-0.5,34.5);
    nvalidTIBHitsH_ = iBook.book1D("nvalidTIBHits", "No. of Valid Hits in Strip TIB",5,-0.5,4.5);
    nvalidTOBHitsH_ = iBook.book1D("nvalidTOBHits", "No. of Valid Hits in Strip TOB",10,-0.5,9.5);
    nvalidTIDHitsH_ = iBook.book1D("nvalidTIDHits", "No. of Valid Hits in Strip TID",5,-0.5,4.5);
    nvalidTECHitsH_ = iBook.book1D("nvalidTECHits", "No. of Valid Hits in Strip TEC",10,-0.5,9.5);

    nlostTrackerHitsH_ = iBook.book1D("nlostTrackerhits", "No. of Lost Tracker Hits",15,-0.5,14.5);
    nlostPixelHitsH_ = iBook.book1D("nlostPixelHits", "No. of Lost Hits in Pixel",5,-0.5,4.5);
    nlostPixelBHitsH_ = iBook.book1D("nlostPixelBarrelHits", "No. of Lost Hits in Pixel Barrel",5,-0.5,4.5);
    nlostPixelEHitsH_ = iBook.book1D("nlostPixelEndcapHits", "No. of Lost Hits in Pixel Endcap",5,-0.5,4.5);
    nlostStripHitsH_ = iBook.book1D("nlostStripHits", "No. of Lost Hits in Strip",10,-0.5,9.5);
    nlostTIBHitsH_ = iBook.book1D("nlostTIBHits", "No. of Lost Hits in Strip TIB",5,-0.5,4.5);
    nlostTOBHitsH_ = iBook.book1D("nlostTOBHits", "No. of Lost Hits in Strip TOB",10,-0.5,9.5);
    nlostTIDHitsH_ = iBook.book1D("nlostTIDHits", "No. of Lost Hits in Strip TID",5,-0.5,4.5);
    nlostTECHitsH_ = iBook.book1D("nlostTECHits", "No. of Lost Hits in Strip TEC",10,-0.5,9.5);    

    trkLayerwithMeasurementH_ = iBook.book1D("trkLayerwithMeasurement", "No. of Layers per Track",20,0.0,20.0);
    pixelLayerwithMeasurementH_ = iBook.book1D("pixelLayerwithMeasurement", "No. of Pixel Layers per Track",10,0.0,10.0);
    pixelBLayerwithMeasurementH_ = iBook.book1D("pixelBLayerwithMeasurement", "No. of Pixel Barrel Layers per Track",5,0.0,5.0);
    pixelELayerwithMeasurementH_ = iBook.book1D("pixelELayerwithMeasurement", "No. of Pixel Endcap Layers per Track",5,0.0,5.0);
    stripLayerwithMeasurementH_ = iBook.book1D("stripLayerwithMeasurement", "No. of Strip Layers per Track",20,0.0,20.0);
    stripTIBLayerwithMeasurementH_ = iBook.book1D("stripTIBLayerwithMeasurement", "No. of Strip TIB Layers per Track",10,0.0,10.0);
    stripTOBLayerwithMeasurementH_ = iBook.book1D("stripTOBLayerwithMeasurement", "No. of Strip TOB Layers per Track",10,0.0,10.0);
    stripTIDLayerwithMeasurementH_ = iBook.book1D("stripTIDLayerwithMeasurement", "No. of Strip TID Layers per Track",5,0.0,5.0);
    stripTECLayerwithMeasurementH_ = iBook.book1D("stripTECLayerwithMeasurement", "No. of Strip TEC Layers per Track",15,0.0,15.0);

    nlostHitsH_ = iBook.book1D("nlostHits", "No. of Lost Hits",10,-0.5,9.5);

    beamSpotXYposH_ = iBook.book1D("beamSpotXYpos", "XY position of beam spot",40,-4.0,4.0);
    beamSpotXYposerrH_ = iBook.book1D("beamSpotXYposerr", "Error in XY position of beam spot",20,0.0,4.0);
    beamSpotZposH_ = iBook.book1D("beamSpotZpos", "Z position of beam spot",100,-20.0,20.0);
    beamSpotZposerrH_ = iBook.book1D("beamSpotZposerr", "Error in Z position of beam spot", 50, 0.0, 5.0);

    vertexXposH_ = iBook.book1D("vertexXpos", "Vertex X position", 50, -1.0, 1.0);
    vertexYposH_ = iBook.book1D("vertexYpos", "Vertex Y position", 50, -1.0, 1.0);
    vertexZposH_ = iBook.book1D("vertexZpos", "Vertex Z position", 100,-20.0,20.0);
    nVertexH_ = iBook.book1D("nVertex", "# of vertices", 60, -0.5, 59.5);
    nVtxH_ = iBook.book1D("nVtx", "# of vtxs", 60, -0.5, 59.5);
    

    nMissingInnerHitBH_ = iBook.book1D("nMissingInnerHitB", "No. missing inner hit per Track in Barrel", 6, -0.5, 5.5);
    nMissingInnerHitEH_ = iBook.book1D("nMissingInnerHitE", "No. missing inner hit per Track in Endcap", 6, -0.5, 5.5);
    nMissingOuterHitBH_ = iBook.book1D("nMissingOuterHitB", "No. missing outer hit per Track in Barrel", 11, -0.5, 10.5);
    nMissingOuterHitEH_ = iBook.book1D("nMissingOuterHitE", "No. missing outer hit per Track in Endcap", 11, -0.5, 10.5);

    residualXPBH_ = iBook.book1D("residualXPixelBarrel", "Residual in X in Pixel Barrel", 20, -5.0, 5.0);
    residualXPEH_ = iBook.book1D("residualXPixelEndcap", "Residual in X in Pixel Endcap", 20, -5.0, 5.0);
    residualXTIBH_ = iBook.book1D("residualXStripTIB", "Residual in X in Strip TIB", 20, -5.0, 5.0);
    residualXTOBH_ = iBook.book1D("residualXStripTOB", "Residual in X in Strip TOB", 20, -5.0, 5.0);
    residualXTECH_ = iBook.book1D("residualXStripTEC", "Residual in X in Strip TEC", 20, -5.0, 5.0);
    residualXTIDH_ = iBook.book1D("residualXStripTID", "Residual in X in Strip TID", 20, -5.0, 5.0);
    residualYPBH_ = iBook.book1D("residualYPixelBarrel", "Residual in Y in Pixel Barrel", 20, -5.0, 5.0);
    residualYPEH_ = iBook.book1D("residualYPixelEndcap", "Residual in Y in Pixel Endcap", 20, -5.0, 5.0);
    residualYTIBH_ = iBook.book1D("residualYStripTIB", "Residual in Y in Strip TIB", 20, -5.0, 5.0);
    residualYTOBH_ = iBook.book1D("residualYStripTOB", "Residual in Y in Strip TOB", 20, -5.0, 5.0);
    residualYTECH_ = iBook.book1D("residualYStripTEC", "Residual in Y in Strip TEC", 20, -5.0, 5.0);
    residualYTIDH_ = iBook.book1D("residualYStripTID", "Residual in Y in Strip TID", 20, -5.0, 5.0);

    nTracksH_ = iBook.book1D("nTracks", "No. of Tracks", 100, -0.5, 999.5);
  }
  if (isMC_) {
    bunchCrossingH_ = iBook.book1D("bunchCrossing", "Bunch Crosssing", 60, 0, 60.0);
    nPUH_ = iBook.book1D("nPU", "No of Pileup", 60, 0, 60.0);
    trueNIntH_ = iBook.book1D("trueNInt", "True no of Interactions", 60, 0, 60.0);
  }
  // Exclusive histograms
  
  nLostHitByLayerH_ = iBook.book1D("nLostHitByLayer", "No. of Lost Hit per Layer", 27, 0.5, 27.5);

  nLostHitByLayerPixH_ = iBook.book1D("nLostHitByLayerPix", "No. of Lost Hit per Layer for Pixel detector", 5, 0.5, 5.5);

  nLostHitByLayerStripH_ = iBook.book1D("nLostHitByLayerStrip", "No. of Lost Hit per Layer for SiStrip detector", 22, 0.5, 22.5);
  
  nLostHitsVspTH_ = iBook.bookProfile("nLostHitsVspT", "Number of Lost Hits Vs pT",
				      TrackPtHistoPar.getParameter<int32_t>("Xbins"),
				      TrackPtHistoPar.getParameter<double>("Xmin"),
				      TrackPtHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsVsEtaH_ = iBook.bookProfile("nLostHitsVsEta", "Number of Lost Hits Vs Eta", 
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"), 
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsVsCosThetaH_ = iBook.bookProfile("nLostHitsVsCosTheta", "Number of Lost Hits Vs Cos(Theta)",50,-1.0,1.0,0.0,0.0,"g");
  nLostHitsVsPhiH_ = iBook.bookProfile("nLostHitsVsPhi", "Number of Lost Hits Vs Phi",100,-3.5,3.5,0.0,0.0,"g");

  nHitsTIBSVsEtaH_ = iBook.bookProfile("nHitsTIBSVsEta", "Number of Hits in TIB Vs Eta (Single-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nHitsTOBSVsEtaH_ = iBook.bookProfile("nHitsTOBSVsEta", "Number of Hits in TOB Vs Eta (Single-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");   
  nHitsTECSVsEtaH_ = iBook.bookProfile("nHitsTECSVsEta", "Number of Hits in TEC Vs Eta (Single-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nHitsTIDSVsEtaH_ = iBook.bookProfile("nHitsTIDSVsEta", "Number of Hits in TID Vs Eta (Single-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  
  nHitsStripSVsEtaH_ = iBook.bookProfile("nHitsStripSVsEta", "Number of Strip Hits Vs Eta (Single-sided)",
					 TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					 TrackEtaHistoPar.getParameter<double>("Xmin"),
					 TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  
  nHitsTIBDVsEtaH_ = iBook.bookProfile("nHitsTIBDVsEta", "Number of Hits in TIB Vs Eta (Double-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nHitsTOBDVsEtaH_ = iBook.bookProfile("nHitsTOBDVsEta", "Number of Hits in TOB Vs Eta (Double-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");   
  nHitsTECDVsEtaH_ = iBook.bookProfile("nHitsTECDVsEta", "Number of Hits in TEC Vs Eta (Double-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nHitsTIDDVsEtaH_ = iBook.bookProfile("nHitsTIDDVsEta", "Number of Hits in TID Vs Eta (Double-sided)",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nHitsStripDVsEtaH_ = iBook.bookProfile("nHitsStripDVsEta", "Number of Strip Hits Vs Eta (Double-sided)",
					 TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					 TrackEtaHistoPar.getParameter<double>("Xmin"),
					 TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  
  nValidHitsVspTH_ = iBook.bookProfile("nValidHitsVspT", "Number of Valid Hits Vs pT",
				  TrackPtHistoPar.getParameter<int32_t>("Xbins"),
				  TrackPtHistoPar.getParameter<double>("Xmin"),
				  TrackPtHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsVsnVtxH_ = iBook.bookProfile("nValidHitsVsnVtx", "Number of Valid Hits Vs Number of Vertex", 100,0,50,0.0,0.0,"g");
  nValidHitsVsEtaH_ = iBook.bookProfile("nValidHitsVsEta", "Number of Hits Vs Eta", 
				   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				   TrackEtaHistoPar.getParameter<double>("Xmin"),
				   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  
  nValidHitsVsCosThetaH_ = iBook.bookProfile("nValidHitsVsCosTheta", "Number of Valid Hits Vs Cos(Theta)", 50,-1.0,1.0,0.0,0.0,"g");
  nValidHitsVsPhiH_ = iBook.bookProfile("nValidHitsVsPhi", "Number of Valid Hits Vs Phi", 100,-3.5,3.5,0.0,0.0,"g");
  
  nValidHitsPixVsEtaH_ = iBook.bookProfile("nValidHitsPixVsEta", "Number of Valid Hits in Pixel Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsPixBVsEtaH_ = iBook.bookProfile("nValidHitsPixBVsEta", "Number of Valid Hits in Pixel Barrel Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsPixEVsEtaH_ = iBook.bookProfile("nValidHitsPixEVsEta", "Number of Valid Hits in Pixel Endcap Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsStripVsEtaH_ = iBook.bookProfile("nValidHitsStripVsEta", "Number of Valid Hits in SiStrip Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTIBVsEtaH_ = iBook.bookProfile("nValidHitsTIBVsEta", "Number of Valid Hits in TIB Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTOBVsEtaH_ = iBook.bookProfile("nValidHitsTOBVsEta", "Number of Valid Hits in TOB Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");   
  nValidHitsTECVsEtaH_ = iBook.bookProfile("nValidHitsTECVsEta", "Number of Valid Hits in TEC Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTIDVsEtaH_ = iBook.bookProfile("nValidHitsTIDVsEta", "Number of Valid Hits in TID Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");

  nValidHitsPixVsPhiH_ = iBook.bookProfile("nValidHitsPixVsPhi", "Number of Valid Hits in Pixel Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsPixBVsPhiH_ = iBook.bookProfile("nValidHitsPixBVsPhi", "Number of Valid Hits in Pixel Barrel Vs Phi",
					    TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					    TrackEtaHistoPar.getParameter<double>("Xmin"),
					    TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsPixEVsPhiH_ = iBook.bookProfile("nValidHitsPixEVsPhi", "Number of Valid Hits in Pixel Endcap Vs Phi",
					    TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					    TrackEtaHistoPar.getParameter<double>("Xmin"),
					    TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsStripVsPhiH_ = iBook.bookProfile("nValidHitsStripVsPhi", "Number of Valid Hits in SiStrip Vs Phi",
					     TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					     TrackEtaHistoPar.getParameter<double>("Xmin"),
					     TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTIBVsPhiH_ = iBook.bookProfile("nValidHitsTIBVsPhi", "Number of Valid Hits in TIB Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTOBVsPhiH_ = iBook.bookProfile("nValidHitsTOBVsPhi", "Number of Valid Hits in TOB Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTECVsPhiH_ = iBook.bookProfile("nValidHitsTECVsPhi", "Number of Valid Hits in TEC Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nValidHitsTIDVsPhiH_ = iBook.bookProfile("nValidHitsTIDVsPhi", "Number of Valid Hits in TID Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");

  nLostHitsPixVsEtaH_ = iBook.bookProfile("nLostHitsPixVsEta", "Number of Lost Hits in Pixel Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsPixBVsEtaH_ = iBook.bookProfile("nLostHitsPixBVsEta", "Number of Lost Hits in Pixel Barrel Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsPixEVsEtaH_ = iBook.bookProfile("nLostHitsPixEVsEta", "Number of Lost Hits in Pixel Endcap Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsStripVsEtaH_ = iBook.bookProfile("nLostHitsStripVsEta", "Number of Lost Hits in SiStrip Vs Eta",
				       TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				       TrackEtaHistoPar.getParameter<double>("Xmin"),
				       TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTIBVsEtaH_ = iBook.bookProfile("nLostHitsTIBVsEta", "Number of Lost Hits in TIB Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTOBVsEtaH_ = iBook.bookProfile("nLostHitsTOBVsEta", "Number of Lost Hits in TOB Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");   
  nLostHitsTECVsEtaH_ = iBook.bookProfile("nLostHitsTECVsEta", "Number of Lost Hits in TEC Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTIDVsEtaH_ = iBook.bookProfile("nLostHitsTIDVsEta", "Number of Lost Hits in TID Vs Eta",
				      TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
				      TrackEtaHistoPar.getParameter<double>("Xmin"),
				      TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");

  nLostHitsPixVsPhiH_ = iBook.bookProfile("nLostHitsPixVsPhi", "Number of Lost Hits in Pixel Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsPixBVsPhiH_ = iBook.bookProfile("nLostHitsPixBVsPhi", "Number of Lost Hits in Pixel Barrel Vs Phi",
					    TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					    TrackEtaHistoPar.getParameter<double>("Xmin"),
					    TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsPixEVsPhiH_ = iBook.bookProfile("nLostHitsPixEVsPhi", "Number of Lost Hits in Pixel Endcap Vs Phi",
					    TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					    TrackEtaHistoPar.getParameter<double>("Xmin"),
					    TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsStripVsPhiH_ = iBook.bookProfile("nLostHitsStripVsPhi", "Number of Lost Hits in SiStrip Vs Phi",
					     TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					     TrackEtaHistoPar.getParameter<double>("Xmin"),
					     TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTIBVsPhiH_ = iBook.bookProfile("nLostHitsTIBVsPhi", "Number of Lost Hits in TIB Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTOBVsPhiH_ = iBook.bookProfile("nLostHitsTOBVsPhi", "Number of Lost Hits in TOB Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTECVsPhiH_ = iBook.bookProfile("nLostHitsTECVsPhi", "Number of Lost Hits in TEC Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");
  nLostHitsTIDVsPhiH_ = iBook.bookProfile("nLostHitsTIDVsPhi", "Number of Lost Hits in TID Vs Phi",
					   TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
					   TrackEtaHistoPar.getParameter<double>("Xmin"),
					   TrackEtaHistoPar.getParameter<double>("Xmax"),0.0,0.0,"g");



  // On and off-track cluster properties
  hOnTrkClusChargeThinH_ = iBook.book1D("hOnTrkClusChargeThin", "On-track Cluster Charge (Thin Sensor)", 100, 0, 1000);
  hOnTrkClusWidthThinH_ = iBook.book1D("hOnTrkClusWidthThin", "On-track Cluster Width (Thin Sensor)", 20, -0.5, 19.5);
  hOnTrkClusChargeThickH_ = iBook.book1D("hOnTrkClusChargeThick", "On-track Cluster Charge (Thick Sensor)", 100, 0, 1000);
  hOnTrkClusWidthThickH_ = iBook.book1D("hOnTrkClusWidthThick", "On-track Cluster Width (Thick Sensor)", 20, -0.5, 19.5);
  
  hOffTrkClusChargeThinH_ = iBook.book1D("hOffTrkClusChargeThin", "Off-track Cluster Charge (Thin Sensor)", 100, 0, 1000);
  hOffTrkClusWidthThinH_ = iBook.book1D("hOffTrkClusWidthThin", "Off-track Cluster Width (Thin Sensor)", 20, -0.5, 19.5);
  hOffTrkClusChargeThickH_ = iBook.book1D("hOffTrkClusChargeThick", "Off-track Cluster Charge (Thick Sensor)", 100, 0, 1000);
  hOffTrkClusWidthThickH_ = iBook.book1D("hOffTrkClusWidthThick", "Off-track Cluster Width (Thick Sensor)", 20, -0.5, 19.5);
}
void StandaloneTrackMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  std::cout << "Begin StandaloneTrackMonitor" << std::endl;

  // Get event setup (to get global transformation)                                  
  edm::ESHandle<TrackerGeometry> geomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  const TrackerGeometry& tkGeom = (*geomHandle);
  std::cout << "Debug level 1" << std::endl;
  
  // Primary vertex collection
  edm::Handle<reco::VertexCollection> vertexColl;
  iEvent.getByToken(vertexToken_, vertexColl);
  std::cout << "Debug level 2" << std::endl;
  if (vertexColl->size() > 0) {
    const reco::Vertex& pv = (*vertexColl)[0];

  // Beam spot
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);
  std::cout << "Debug level 3" << std::endl;

  // Track collection
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackToken_, tracks);
  std::cout << "Debug level 4" << std::endl;  

  // Access PU information
  double wfac = 1.0;  // for data
  if (!iEvent.isRealData()) {
    edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
    iEvent.getByToken(puSummaryToken_, PupInfo);
    
    if (verbose_) edm::LogInfo("StandaloneTrackMonitor") << "nPUColl = " << PupInfo->size();
    for (auto const& v : *PupInfo) {
      int bx = v.getBunchCrossing();
      if (bunchCrossingH_) bunchCrossingH_->Fill(bx);
      if (bx == 0) {
        if (nPUH_) nPUH_->Fill(v.getPU_NumInteractions());
        int ntrueInt = v.getTrueNumInteractions();
	int nVertex = (vertexColl.isValid() ? vertexColl->size() : 0);
        if (trueNIntH_) trueNIntH_->Fill(ntrueInt);
        if (doPUCorrection_) 
          if (nVertex > -1 && nVertex < int(vpu_.size())) wfac = vpu_.at(nVertex);
	  //if (ntrueInt > -1 && ntrueInt < int(vpu_.size())) wfac = vpu_.at(ntrueInt);
      }
    }
  }
  if (verbose_) edm::LogInfo("StandaloneTrackMonitor") << "PU reweight factor = " << wfac;
  std::cout << "PU scale factor" << wfac << std::endl;

  if (!vertexColl.isValid())
    edm::LogError("DqmTrackStudy") << "Error! Failed to get reco::Vertex Collection, " << vertexTag_;
  std::cout << "Debug level 5" << std::endl;

  //int nvtx = (vertexColl.isValid() ? vertexColl->size() : 0);
  //if (nvtx < 10 || nvtx > 30) return;
  if (haveAllHistograms_) {
    int nvtx = (vertexColl.isValid() ? vertexColl->size() : 0);
    nVertexH_->Fill(nvtx, wfac);
    nVtxH_->Fill(nvtx);
  }
  
  std::cout << "Debug level 6" << std::endl;

  int ntracks = 0;
  if (tracks.isValid()) {
    edm::LogInfo("StandaloneTrackMonitor") << "Total # of Tracks: " << tracks->size();
    if (verbose_) edm::LogInfo("StandaloneTrackMonitor") <<"Total # of Tracks: " << tracks->size();
    reco::Track::TrackQuality quality = reco::Track::qualityByName(trackQuality_);
    std::cout <<"Total # of Tracks: " << tracks->size() << std::endl;

    for (auto const& track : *tracks) {
      if (!track.quality(quality)) continue;
      //std::cout << "Debug level 7" << std::endl;
      ++ntracks;     
 
      double eta = track.eta();
      double theta = track.theta();
      double phi = track.phi();
      double pt = track.pt();

      const reco::HitPattern& hitp = track.hitPattern();
      double nValidTrackerHits = hitp.numberOfValidTrackerHits();
      double nValidPixelHits = hitp.numberOfValidPixelHits();
      double nValidPixelBHits = hitp.numberOfValidPixelBarrelHits();
      double nValidPixelEHits = hitp.numberOfValidPixelEndcapHits();
      double nValidStripHits = hitp.numberOfValidStripHits();
      double nValidTIBHits = hitp.numberOfValidStripTIBHits();
      double nValidTOBHits = hitp.numberOfValidStripTOBHits();
      double nValidTIDHits = hitp.numberOfValidStripTIDHits();
      double nValidTECHits = hitp.numberOfValidStripTECHits();

      int missingInnerHit = hitp.numberOfHits(reco::HitPattern::MISSING_INNER_HITS);
      int missingOuterHit = hitp.numberOfHits(reco::HitPattern::MISSING_OUTER_HITS);

      nValidHitsVspTH_->Fill(pt, nValidTrackerHits);
      nValidHitsVsEtaH_->Fill(eta, nValidTrackerHits);
      nValidHitsVsCosThetaH_->Fill(std::cos(theta), nValidTrackerHits);
      nValidHitsVsPhiH_->Fill(phi, nValidTrackerHits);
      nValidHitsVsnVtxH_->Fill(vertexColl->size(), nValidTrackerHits);

      nValidHitsPixVsEtaH_->Fill(eta, nValidPixelHits);
      nValidHitsPixBVsEtaH_->Fill(eta, nValidPixelBHits);
      nValidHitsPixEVsEtaH_->Fill(eta, nValidPixelEHits);
      nValidHitsStripVsEtaH_->Fill(eta, nValidStripHits);
      nValidHitsTIBVsEtaH_->Fill(eta, nValidTIBHits);
      nValidHitsTOBVsEtaH_->Fill(eta, nValidTOBHits);
      nValidHitsTECVsEtaH_->Fill(eta, nValidTECHits);
      nValidHitsTIDVsEtaH_->Fill(eta, nValidTIDHits);

      nValidHitsPixVsPhiH_->Fill(phi, nValidPixelHits);
      nValidHitsPixBVsPhiH_->Fill(phi, nValidPixelBHits);
      nValidHitsPixEVsPhiH_->Fill(phi, nValidPixelEHits);
      nValidHitsStripVsPhiH_->Fill(phi, nValidStripHits);
      nValidHitsTIBVsPhiH_->Fill(phi, nValidTIBHits);
      nValidHitsTOBVsPhiH_->Fill(phi, nValidTOBHits);
      nValidHitsTECVsPhiH_->Fill(phi, nValidTECHits);
      nValidHitsTIDVsPhiH_->Fill(phi, nValidTIDHits);

      int nLostHits = track.numberOfLostHits();
      int nLostTrackerHits = hitp.numberOfLostTrackerHits(reco::HitPattern::TRACK_HITS);
      int nLostPixHits = hitp.numberOfLostPixelHits(reco::HitPattern::TRACK_HITS);
      int nLostPixBHits = hitp.numberOfLostPixelBarrelHits(reco::HitPattern::TRACK_HITS);
      int nLostPixEHits = hitp.numberOfLostPixelEndcapHits(reco::HitPattern::TRACK_HITS);
      int nLostStripHits = hitp.numberOfLostStripHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTIBHits = hitp.numberOfLostStripTIBHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTIDHits = hitp.numberOfLostStripTIDHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTOBHits = hitp.numberOfLostStripTOBHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTECHits = hitp.numberOfLostStripTECHits(reco::HitPattern::TRACK_HITS);
      
      nLostHitsVspTH_->Fill(pt, nLostTrackerHits);
      nLostHitsVsEtaH_->Fill(eta, nLostTrackerHits);
      nLostHitsVsCosThetaH_->Fill(std::cos(theta), nLostTrackerHits);
      nLostHitsVsPhiH_->Fill(phi, nLostTrackerHits);

      nLostHitsPixVsEtaH_->Fill(eta, nLostPixHits);
      nLostHitsPixBVsEtaH_->Fill(eta, nLostPixBHits);
      nLostHitsPixEVsEtaH_->Fill(eta, nLostPixEHits);
      nLostHitsStripVsEtaH_->Fill(eta, nLostStripHits);
      nLostHitsTIBVsEtaH_->Fill(eta, nLostStripTIBHits);
      nLostHitsTOBVsEtaH_->Fill(eta, nLostStripTOBHits);
      nLostHitsTECVsEtaH_->Fill(eta, nLostStripTECHits);
      nLostHitsTIDVsEtaH_->Fill(eta, nLostStripTIDHits);

      nLostHitsPixVsPhiH_->Fill(phi, nLostPixHits);
      nLostHitsPixBVsPhiH_->Fill(phi, nLostPixBHits);
      nLostHitsPixEVsPhiH_->Fill(phi, nLostPixEHits);
      nLostHitsStripVsPhiH_->Fill(phi, nLostStripHits);
      nLostHitsTIBVsPhiH_->Fill(phi, nLostStripTIBHits);
      nLostHitsTOBVsPhiH_->Fill(phi, nLostStripTOBHits);
      nLostHitsTECVsPhiH_->Fill(phi, nLostStripTECHits);
      nLostHitsTIDVsPhiH_->Fill(phi, nLostStripTIDHits);


      if (abs(eta) <= 1.4) {
	nMissingInnerHitBH_->Fill(missingInnerHit);
	nMissingOuterHitBH_->Fill(missingOuterHit);
      }
      else {
	nMissingInnerHitEH_->Fill(missingInnerHit);
        nMissingOuterHitEH_->Fill(missingOuterHit);
      }
      double residualXPB = 0, residualXPE = 0, residualXTIB = 0, residualXTOB = 0, residualXTEC = 0, residualXTID = 0;
      double residualYPB = 0, residualYPE = 0, residualYTIB = 0, residualYTOB = 0, residualYTEC = 0, residualYTID = 0; 
      reco::TrackResiduals residuals = track.residuals();
      int i = 0;
      for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it,++i) {
        const TrackingRecHit& hitn = (**it);
	if (hitn.isValid()) {
	  int subdet = hitn.geographicalId().subdetId();

	  if      (subdet == PixelSubdetector::PixelBarrel) {
	    residualXPB  = residuals.residualX(i, hitp);
	    residualYPB  = residuals.residualY(i, hitp);
	  }
	  else if (subdet == PixelSubdetector::PixelEndcap) {
	    residualXPE  = residuals.residualX(i, hitp);
	    residualYPE  = residuals.residualY(i, hitp);
	  }
	  else if (subdet == StripSubdetector::TIB) {
	    residualXTIB = residuals.residualX(i, hitp);
	    residualYTIB = residuals.residualY(i, hitp);
	  }
	  else if (subdet == StripSubdetector::TOB) {
	    residualXTOB = residuals.residualX(i, hitp);
	    residualYTOB = residuals.residualY(i, hitp);
	  }
	  else if (subdet == StripSubdetector::TEC) {
	    residualXTEC = residuals.residualX(i, hitp);
	    residualYTEC = residuals.residualY(i, hitp);
	  }
	  else if (subdet == StripSubdetector::TID) {
	    residualXTID = residuals.residualX(i, hitp);
	    residualYTID = residuals.residualY(i, hitp);
	  }
	}	
	residualXPBH_->Fill(residualXPB);
	residualXPEH_->Fill(residualXPE);
	residualXTIBH_->Fill(residualXTIB);
	residualXTOBH_->Fill(residualXTOB);
	residualXTECH_->Fill(residualXTEC);
	residualXTIDH_->Fill(residualXTID);
        residualYPBH_->Fill(residualYPB);
        residualYPEH_->Fill(residualYPE);
        residualYTIBH_->Fill(residualYTIB);
        residualYTOBH_->Fill(residualYTOB);
        residualYTECH_->Fill(residualYTEC);
        residualYTIDH_->Fill(residualYTID);
	}
      for (int i = 0; i < hitp.numberOfHits(reco::HitPattern::TRACK_HITS); i++) {
	uint32_t hit = hitp.getHitPattern(reco::HitPattern::TRACK_HITS, i);
	if (hitp.missingHitFilter(hit)) {
	  double losthitBylayer = -1.0;
	  double losthitBylayerPix = -1.0;
	  double losthitBylayerStrip = -1.0;
	  int layer = hitp.getLayer(hit);
	  //int side = hitp.getSide(hit);
	  if (hitp.pixelBarrelHitFilter(hit)) {
	    losthitBylayer = layer;
	    losthitBylayerPix = layer;	    
	  }
	  else if (hitp.pixelEndcapHitFilter(hit)) {
	    //losthitBylayer = (side == 0) ? layer+3 : layer+5;
	    losthitBylayer = layer+3;
	    losthitBylayerPix = layer+3;
	  }
	  else if (hitp.stripTIBHitFilter(hit)) {
	    losthitBylayer = layer + 5;
	    losthitBylayerStrip = layer;
	  }
	  else if (hitp.stripTIDHitFilter(hit)) {
	    //losthitBylayer = (side == 0) ? layer+11 : layer+14;
	    losthitBylayer = layer+9;
	    losthitBylayerStrip = layer+4;
	  }
	  else if (hitp.stripTOBHitFilter(hit)) {
	    losthitBylayer = layer + 12;
	    losthitBylayerStrip = layer + 7;
	  }
	  else if (hitp.stripTECHitFilter(hit)) {
	    //losthitBylayer = (side == 0) ? layer+23 : layer+32;
	    losthitBylayer = layer+18;
	    losthitBylayerStrip = layer+13;
	  }
	  if (losthitBylayer > -1) nLostHitByLayerH_->Fill(losthitBylayer, wfac);
	  if (losthitBylayerPix > -1) nLostHitByLayerPixH_->Fill(losthitBylayerPix, wfac);
	  if (losthitBylayerStrip > -1) nLostHitByLayerStripH_->Fill(losthitBylayerStrip, wfac);
	}
      }
      
      if (haveAllHistograms_) {
	double etaError = track.etaError();
	double thetaError = track.thetaError();
	double phiError = track.phiError();
	double p = track.p();
	double ptError = track.ptError();
	double qoverp = track.qoverp();
	double qoverpError = track.qoverpError();
	double charge = track.charge();
	
	double dxy = track.dxy(beamSpot->position());
	double dxyError = track.dxyError();
	double dz = track.dz(beamSpot->position());
	double dzError = track.dzError();
	
	double trkd0 = track.d0();     
	double chi2 = track.chi2();
	double ndof = track.ndof();
	double chi2prob = TMath::Prob(track.chi2(),(int)track.ndof());
	double chi2oNDF = track.normalizedChi2();
	double vx = track.vx();
	double vy = track.vy();
	double vz = track.vz();
	
 	double distanceOfClosestApproachToPV = track.dxy(pv.position());
	double xPointOfClosestApproachwrtPV = track.vx()-pv.position().x();
	double yPointOfClosestApproachwrtPV = track.vy()-pv.position().y();
	double positionZ0 = track.dz(pv.position());
				     
	edm::ESHandle<TransientTrackBuilder> theB;
	iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
	reco::TransientTrack transTrack = theB->build(track);

	double sip3dToPV = 0, sip2dToPV = 0;
	GlobalVector dir(track.px(), track.py(), track.pz());
	std::pair<bool, Measurement1D> ip3d = IPTools::signedImpactParameter3D(transTrack, dir, pv);
	std::pair<bool, Measurement1D> ip2d = IPTools::signedTransverseImpactParameter(transTrack, dir, pv);
	if(ip3d.first) sip3dToPV = ip3d.second.value()/ip3d.second.error();
	if(ip2d.first) sip2dToPV = ip2d.second.value()/ip2d.second.error();
        double sipDxyToPV = track.dxy(pv.position())/track.dxyError();
	double sipDzToPV = track.dz(pv.position())/track.dzError();				             
	
	// Fill the histograms
	trackEtaH_->Fill(eta, wfac);
	trackEtaerrH_->Fill(etaError, wfac);
	trackCosThetaH_->Fill(std::cos(theta), wfac);
	trackThetaerrH_->Fill(thetaError, wfac);
	trackPhiH_->Fill(phi, wfac);
	trackPhierrH_->Fill(phiError, wfac);
	trackPH_->Fill(p, wfac);
	trackPtH_->Fill(pt, wfac);
	if (pt <= 2) trackPtUpto2GeVH_->Fill(pt, wfac);
	if (pt >= 10) trackPtOver10GeVH_->Fill(pt, wfac);
	trackPterrH_->Fill(ptError, wfac);
	trackqOverpH_->Fill(qoverp, wfac);
	trackqOverperrH_->Fill(qoverpError, wfac);
	trackChargeH_->Fill(charge, wfac);
	trackChi2H_->Fill(chi2, wfac);
	trackChi2ProbH_->Fill(chi2prob, wfac);
	trackChi2oNDFH_->Fill(chi2oNDF, wfac);
	trackd0H_->Fill(trkd0, wfac);
        tracknDOFH_->Fill(ndof, wfac);
	trackChi2bynDOFH_->Fill(chi2/ndof, wfac);

	nlostHitsH_->Fill(nLostHits, wfac);
	nlostTrackerHitsH_->Fill(nLostTrackerHits, wfac);

	beamSpotXYposH_->Fill(dxy, wfac);
	beamSpotXYposerrH_->Fill(dxyError, wfac);
	beamSpotZposH_->Fill(dz, wfac);
	beamSpotZposerrH_->Fill(dzError, wfac);

	vertexXposH_->Fill(vx, wfac);
	vertexYposH_->Fill(vy, wfac);
	vertexZposH_->Fill(vz, wfac);

        DistanceOfClosestApproachToPVH_->Fill(distanceOfClosestApproachToPV);
	DistanceOfClosestApproachToPVVsPhiH_->Fill(phi, distanceOfClosestApproachToPV);
	xPointOfClosestApproachVsZ0wrtPVH_->Fill(positionZ0, xPointOfClosestApproachwrtPV);
	yPointOfClosestApproachVsZ0wrtPVH_->Fill(positionZ0, yPointOfClosestApproachwrtPV);

	sip3dToPVH_->Fill(sip3dToPV);
	sip2dToPVH_->Fill(sip2dToPV);
	sipDxyToPVH_->Fill(sipDxyToPV);
	sipDzToPVH_->Fill(sipDzToPV);
       			     
        double trackerLayersWithMeasurement = hitp.trackerLayersWithMeasurement();
	double pixelLayersWithMeasurement = hitp.pixelLayersWithMeasurement();
	double pixelBLayersWithMeasurement = hitp.pixelBarrelLayersWithMeasurement();
	double pixelELayersWithMeasurement = hitp.pixelEndcapLayersWithMeasurement();
	double stripLayersWithMeasurement = hitp.stripLayersWithMeasurement();
	double stripTIBLayersWithMeasurement = hitp.stripTIBLayersWithMeasurement();
	double stripTOBLayersWithMeasurement = hitp.stripTOBLayersWithMeasurement();
	double stripTIDLayersWithMeasurement = hitp.stripTIDLayersWithMeasurement();
	double stripTECLayersWithMeasurement = hitp.stripTECLayersWithMeasurement();
	
	trkLayerwithMeasurementH_->Fill(trackerLayersWithMeasurement, wfac);
	pixelLayerwithMeasurementH_->Fill(pixelLayersWithMeasurement, wfac);
	pixelBLayerwithMeasurementH_->Fill(pixelBLayersWithMeasurement, wfac);
	pixelELayerwithMeasurementH_->Fill(pixelELayersWithMeasurement, wfac);
	stripLayerwithMeasurementH_->Fill(stripLayersWithMeasurement, wfac);
	stripTIBLayerwithMeasurementH_->Fill(stripTIBLayersWithMeasurement, wfac);
	stripTOBLayerwithMeasurementH_->Fill(stripTOBLayersWithMeasurement, wfac);
	stripTIDLayerwithMeasurementH_->Fill(stripTIDLayersWithMeasurement, wfac);
	stripTECLayerwithMeasurementH_->Fill(stripTECLayersWithMeasurement, wfac);

	nvalidTrackerHitsH_->Fill(nValidTrackerHits, wfac);
	nvalidPixelHitsH_->Fill(nValidPixelHits, wfac);
	nvalidPixelBHitsH_->Fill(nValidPixelBHits, wfac);
	nvalidPixelEHitsH_->Fill(nValidPixelEHits, wfac);
	nvalidStripHitsH_->Fill(nValidStripHits, wfac);
	nvalidTIBHitsH_->Fill(nValidTIBHits, wfac);
	nvalidTOBHitsH_->Fill(nValidTOBHits, wfac);
	nvalidTIDHitsH_->Fill(nValidTIDHits, wfac);
	nvalidTECHitsH_->Fill(nValidTECHits, wfac);

	nlostTrackerHitsH_->Fill(nLostTrackerHits, wfac);
	nlostPixelHitsH_->Fill(nLostPixHits, wfac);
	nlostPixelBHitsH_->Fill(nLostPixBHits, wfac);
	nlostPixelEHitsH_->Fill(nLostPixEHits, wfac);
	nlostStripHitsH_->Fill(nLostStripHits, wfac);
	nlostTIBHitsH_->Fill(nLostStripTIBHits, wfac);
	nlostTOBHitsH_->Fill(nLostStripTOBHits, wfac);
	nlostTIDHitsH_->Fill(nLostStripTIDHits, wfac);
	nlostTECHitsH_->Fill(nLostStripTECHits, wfac);
      }
      int nStripTIBS = 0, nStripTOBS = 0, nStripTECS = 0, nStripTIDS = 0;
      int nStripTIBD = 0, nStripTOBD = 0, nStripTECD = 0, nStripTIDD = 0;
      for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
        const TrackingRecHit& hit = (**it);
	if (hit.isValid()) {
	  if (hit.geographicalId().det() == DetId::Tracker) {
	    int subdetId = hit.geographicalId().subdetId();
            
            // Find on-track clusters
            processHit(hit, iSetup, tkGeom, wfac);
	    
	    const DetId detId(hit.geographicalId());
	    const SiStripDetId stripId(detId);
	    if (0) std::cout << "Hit Dimension: " << hit.dimension()
			     << ", isGlued: " << stripId.glued()
			     << ", isStereo: " << stripId.stereo()
			     << std::endl;
	    
	    if (stripId.glued()) {
	      if      (subdetId == StripSubdetector::TIB) ++nStripTIBD;
	      else if (subdetId == StripSubdetector::TOB) ++nStripTOBD;
	      else if (subdetId == StripSubdetector::TEC) ++nStripTECD;
	      else if (subdetId == StripSubdetector::TID) ++nStripTIDD;
	    }
	    else {
	      if      (subdetId == StripSubdetector::TIB) ++nStripTIBS;
	      else if (subdetId == StripSubdetector::TOB) ++nStripTOBS;
	      else if (subdetId == StripSubdetector::TEC) ++nStripTECS;
	      else if (subdetId == StripSubdetector::TID) ++nStripTIDS;
	    }
	  }
	}
      }   
      
      nHitsTIBSVsEtaH_->Fill(eta, nStripTIBS);
      nHitsTOBSVsEtaH_->Fill(eta, nStripTOBS);
      nHitsTECSVsEtaH_->Fill(eta, nStripTECS);
      nHitsTIDSVsEtaH_->Fill(eta, nStripTIDS);
      nHitsStripSVsEtaH_->Fill(eta, nStripTIBS+nStripTOBS+nStripTECS+nStripTIDS);
      
      nHitsTIBDVsEtaH_->Fill(eta, nStripTIBD);
      nHitsTOBDVsEtaH_->Fill(eta, nStripTOBD);
      nHitsTECDVsEtaH_->Fill(eta, nStripTECD);
      nHitsTIDDVsEtaH_->Fill(eta, nStripTIDD);
      nHitsStripDVsEtaH_->Fill(eta, nStripTIBD+nStripTOBD+nStripTECD+nStripTIDD);
    }
    std::cout << "Debug level 8" << std::endl;
  } 
  else {
    edm::LogError("DqmTrackStudy") << "Error! Failed to get reco::Track collection, " << trackTag_;
  }
  if (haveAllHistograms_) nTracksH_->Fill(ntracks, wfac);
  
  // off track cluster properties
  processClusters(iEvent, iSetup, tkGeom, wfac);
  
  std::cout << "Ends StandaloneTrackMonitor successfully" << std::endl;
  }
}
void StandaloneTrackMonitor::processClusters(edm::Event const& iEvent, edm::EventSetup const& iSetup, const TrackerGeometry& tkGeom, double wfac)
{
  // SiStripClusters
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterHandle;
  iEvent.getByToken(clusterToken_, clusterHandle);

  if (clusterHandle.isValid()) {
    // Loop on Dets
    for (edmNew::DetSetVector<SiStripCluster>::const_iterator dsvit  = clusterHandle->begin(); 
                                                              dsvit != clusterHandle->end();
	                                                    ++dsvit)
    {
      uint32_t detId = dsvit->id();
      std::map<uint32_t, std::set<const SiStripCluster*> >::iterator jt = clusterMap_.find(detId);
      bool detid_found = (jt != clusterMap_.end()) ? true : false;

      // Loop on Clusters
      for (edmNew::DetSet<SiStripCluster>::const_iterator clusit  = dsvit->begin(); 
                                                          clusit != dsvit->end();	  
                                                        ++clusit)
      {
	if (detid_found) {
	  std::set<const SiStripCluster*>& s = jt->second;
          if (s.find(&*clusit) != s.end()) continue;
	}

        SiStripClusterInfo info(*clusit, iSetup, detId);
        float charge = info.charge();
        float width = info.width();

	const GeomDetUnit* detUnit = tkGeom.idToDetUnit(detId);
	float thickness =  detUnit->surface().bounds().thickness(); // unit cm
        if (thickness > 0.035) {
          hOffTrkClusChargeThickH_->Fill(charge, wfac);
          hOffTrkClusWidthThickH_->Fill(width, wfac);
        }
        else {
	  hOffTrkClusChargeThinH_->Fill(charge, wfac);
	  hOffTrkClusWidthThinH_->Fill(width, wfac);
        }
      }
    }
  }
  else {
    edm::LogError("StandaloneTrackMonitor") << "ClusterCollection " << clusterTag_ << " not valid!!" << std::endl;
  }
}
void StandaloneTrackMonitor::processHit(const TrackingRecHit& recHit, edm::EventSetup const& iSetup, const TrackerGeometry& tkGeom, double wfac)
{
  uint32_t detid = recHit.geographicalId();
  const GeomDetUnit* detUnit = tkGeom.idToDetUnit(detid);
  float thickness =  detUnit->surface().bounds().thickness(); // unit cm      

  auto const& thit = static_cast<BaseTrackerRecHit const&>(recHit);
  if (!thit.isValid()) return;

  auto const& clus = thit.firstClusterRef();
  if (!clus.isValid()) return;
  if (!clus.isStrip()) return;

  if (thit.isMatched()) {
    const SiStripMatchedRecHit2D& matchedHit = dynamic_cast<const SiStripMatchedRecHit2D&>(recHit);

    auto& clusterM = matchedHit.monoCluster();
    SiStripClusterInfo infoM(clusterM, iSetup, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(infoM.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(infoM.width(), wfac);
    }
    else {
      hOnTrkClusChargeThinH_->Fill(infoM.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(infoM.width(), wfac);
    }
    addClusterToMap(detid, &clusterM);

    auto& clusterS = matchedHit.stereoCluster();
    SiStripClusterInfo infoS(clusterS, iSetup, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(infoS.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(infoS.width(), wfac );
    }
    else {
      hOnTrkClusChargeThinH_->Fill(infoS.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(infoS.width(), wfac);
    }
    addClusterToMap(detid, &clusterS);
  }
  else {
    auto& cluster = clus.stripCluster();
    SiStripClusterInfo info(cluster, iSetup, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(info.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(info.width(), wfac);
    }
    else {
      hOnTrkClusChargeThinH_->Fill(info.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(info.width(), wfac);
    }
    addClusterToMap(detid, &cluster);
  }
}
void StandaloneTrackMonitor::addClusterToMap(uint32_t detid, const SiStripCluster* cluster) {
  std::map<uint32_t, std::set<const SiStripCluster*> >::iterator it = clusterMap_.find(detid);
  if (it == clusterMap_.end()) {
    std::set<const SiStripCluster*> s;
    s.insert(cluster);
    clusterMap_.insert(std::pair<uint32_t, std::set<const SiStripCluster*> >(detid, s));
  }
  else {
    std::set<const SiStripCluster*>& s = it->second;
    s.insert(cluster);
  }
}
void StandaloneTrackMonitor::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup){
}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StandaloneTrackMonitor);

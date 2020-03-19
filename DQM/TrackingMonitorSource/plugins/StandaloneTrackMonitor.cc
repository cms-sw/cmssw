#include "TFile.h"
#include "TH1.h"
#include "TMath.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TPRegexp.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DQM/TrackingMonitorSource/interface/StandaloneTrackMonitor.h"
// -----------------------------
// constructors and destructor
// -----------------------------
StandaloneTrackMonitor::StandaloneTrackMonitor(const edm::ParameterSet& ps)
    : parameters_(ps),
      moduleName_(parameters_.getUntrackedParameter<std::string>("moduleName", "StandaloneTrackMonitor")),
      folderName_(parameters_.getUntrackedParameter<std::string>("folderName", "highPurityTracks")),
      trackTag_(parameters_.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      bsTag_(parameters_.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      vertexTag_(
          parameters_.getUntrackedParameter<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"))),
      puSummaryTag_(parameters_.getUntrackedParameter<edm::InputTag>("puTag", edm::InputTag("addPileupInfo"))),
      clusterTag_(parameters_.getUntrackedParameter<edm::InputTag>("clusterTag", edm::InputTag("siStripClusters"))),
      trackToken_(consumes<reco::TrackCollection>(trackTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      vertexToken_(consumes<reco::VertexCollection>(vertexTag_)),
      puSummaryToken_(consumes<std::vector<PileupSummaryInfo> >(puSummaryTag_)),
      clusterToken_(consumes<edmNew::DetSetVector<SiStripCluster> >(clusterTag_)),
      trackQuality_(parameters_.getUntrackedParameter<std::string>("trackQuality", "highPurity")),
      siStripClusterInfo_(consumesCollector()),
      doPUCorrection_(parameters_.getUntrackedParameter<bool>("doPUCorrection", false)),
      isMC_(parameters_.getUntrackedParameter<bool>("isMC", false)),
      haveAllHistograms_(parameters_.getUntrackedParameter<bool>("haveAllHistograms", false)),
      puScaleFactorFile_(
          parameters_.getUntrackedParameter<std::string>("puScaleFactorFile", "PileupScaleFactor_run203002.root")),
      verbose_(parameters_.getUntrackedParameter<bool>("verbose", false)) {
  trackEtaH_ = nullptr;
  trackEtaerrH_ = nullptr;
  trackCosThetaH_ = nullptr;
  trackThetaerrH_ = nullptr;
  trackPhiH_ = nullptr;
  trackPhierrH_ = nullptr;
  trackPH_ = nullptr;
  trackPtH_ = nullptr;
  trackPtUpto2GeVH_ = nullptr;
  trackPtOver10GeVH_ = nullptr;
  trackPterrH_ = nullptr;
  trackqOverpH_ = nullptr;
  trackqOverperrH_ = nullptr;
  trackChargeH_ = nullptr;
  nlostHitsH_ = nullptr;
  nvalidTrackerHitsH_ = nullptr;
  nvalidPixelHitsH_ = nullptr;
  nvalidStripHitsH_ = nullptr;
  trkLayerwithMeasurementH_ = nullptr;
  pixelLayerwithMeasurementH_ = nullptr;
  stripLayerwithMeasurementH_ = nullptr;
  beamSpotXYposH_ = nullptr;
  beamSpotXYposerrH_ = nullptr;
  beamSpotZposH_ = nullptr;
  beamSpotZposerrH_ = nullptr;
  trackChi2H_ = nullptr;
  tracknDOFH_ = nullptr;
  trackd0H_ = nullptr;
  trackChi2bynDOFH_ = nullptr;
  vertexXposH_ = nullptr;
  vertexYposH_ = nullptr;
  vertexZposH_ = nullptr;

  nPixBarrelH_ = nullptr;
  nPixEndcapH_ = nullptr;
  nStripTIBH_ = nullptr;
  nStripTOBH_ = nullptr;
  nStripTECH_ = nullptr;
  nStripTIDH_ = nullptr;
  nTracksH_ = nullptr;

  // for MC only
  nVertexH_ = nullptr;
  bunchCrossingH_ = nullptr;
  nPUH_ = nullptr;
  trueNIntH_ = nullptr;

  nHitsVspTH_ = nullptr;
  nHitsVsEtaH_ = nullptr;
  nHitsVsCosThetaH_ = nullptr;
  nHitsVsPhiH_ = nullptr;
  nHitsVsnVtxH_ = nullptr;
  nLostHitsVspTH_ = nullptr;
  nLostHitsVsEtaH_ = nullptr;
  nLostHitsVsCosThetaH_ = nullptr;
  nLostHitsVsPhiH_ = nullptr;

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
    for (int i = 1; i <= h1->GetNbinsX(); ++i)
      vpu_.push_back(h1->GetBinContent(i));
    f1->Close();
  }
}

void StandaloneTrackMonitor::bookHistograms(DQMStore::IBooker& iBook,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  edm::ParameterSet TrackEtaHistoPar = parameters_.getParameter<edm::ParameterSet>("trackEtaH");
  edm::ParameterSet TrackPtHistoPar = parameters_.getParameter<edm::ParameterSet>("trackPtH");

  std::string currentFolder = moduleName_ + "/" + folderName_;
  iBook.setCurrentFolder(currentFolder);

  // The following are common with the official tool
  if (haveAllHistograms_) {
    if (!trackEtaH_)
      trackEtaH_ = iBook.book1D("trackEta",
                                "Track Eta",
                                TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                TrackEtaHistoPar.getParameter<double>("Xmin"),
                                TrackEtaHistoPar.getParameter<double>("Xmax"));
    if (!trackEtaerrH_)
      trackEtaerrH_ = iBook.book1D("trackEtaerr", "Track Eta Error", 50, 0.0, 1.0);
    if (!trackCosThetaH_)
      trackCosThetaH_ = iBook.book1D("trackCosTheta", "Track Cos(Theta)", 50, -1.0, 1.0);
    if (!trackThetaerrH_)
      trackThetaerrH_ = iBook.book1D("trackThetaerr", "Track Theta Error", 50, 0.0, 1.0);
    if (!trackPhiH_)
      trackPhiH_ = iBook.book1D("trackPhi", "Track Phi", 70, -3.5, 3.5);
    if (!trackPhierrH_)
      trackPhierrH_ = iBook.book1D("trackPhierr", "Track Phi Error", 50, 0.0, 1.0);

    if (!trackPH_)
      trackPH_ = iBook.book1D("trackP", "Track 4-momentum", 50, 0.0, 10.0);
    if (!trackPtH_)
      trackPtH_ = iBook.book1D("trackPt",
                               "Track Pt",
                               TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                               TrackPtHistoPar.getParameter<double>("Xmin"),
                               TrackPtHistoPar.getParameter<double>("Xmax"));
    if (!trackPtUpto2GeVH_)
      trackPtUpto2GeVH_ = iBook.book1D("trackPtUpto2GeV", "Track Pt upto 2GeV", 100, 0, 2.0);
    if (!trackPtOver10GeVH_)
      trackPtOver10GeVH_ = iBook.book1D("trackPtOver10GeV", "Track Pt greater than 10 GeV", 100, 0, 100.0);
    if (!trackPterrH_)
      trackPterrH_ = iBook.book1D("trackPterr", "Track Pt Error", 100, 0.0, 100.0);
    if (!trackqOverpH_)
      trackqOverpH_ = iBook.book1D("trackqOverp", "q Over p", 40, -10.0, 10.0);
    if (!trackqOverperrH_)
      trackqOverperrH_ = iBook.book1D("trackqOverperr", "q Over p Error", 50, 0.0, 25.0);
    if (!trackChargeH_)
      trackChargeH_ = iBook.book1D("trackCharge", "Track Charge", 50, -5, 5);
    if (!trackChi2H_)
      trackChi2H_ = iBook.book1D("trackChi2", "Chi2", 100, 0.0, 100.0);
    if (!tracknDOFH_)
      tracknDOFH_ = iBook.book1D("tracknDOF", "nDOF", 100, 0.0, 100.0);
    if (!trackd0H_)
      trackd0H_ = iBook.book1D("trackd0", "Track d0", 100, -1, 1);
    if (!trackChi2bynDOFH_)
      trackChi2bynDOFH_ = iBook.book1D("trackChi2bynDOF", "Chi2 Over nDOF", 100, 0.0, 10.0);

    if (!nlostHitsH_)
      nlostHitsH_ = iBook.book1D("nlostHits", "No. of Lost Hits", 10, 0.0, 10.0);
    if (!nvalidTrackerHitsH_)
      nvalidTrackerHitsH_ = iBook.book1D("nvalidTrackerhits", "No. of Valid Tracker Hits", 35, 0.0, 35.0);
    if (!nvalidPixelHitsH_)
      nvalidPixelHitsH_ = iBook.book1D("nvalidPixelHits", "No. of Valid Hits in Pixel", 10, 0.0, 10.0);
    if (!nvalidStripHitsH_)
      nvalidStripHitsH_ = iBook.book1D("nvalidStripHits", "No.of Valid Hits in Strip", 25, 0.0, 25.0);

    if (!trkLayerwithMeasurementH_)
      trkLayerwithMeasurementH_ = iBook.book1D("trkLayerwithMeasurement", "No. of Layers per Track", 25, 0.0, 25.0);
    if (!pixelLayerwithMeasurementH_)
      pixelLayerwithMeasurementH_ =
          iBook.book1D("pixelLayerwithMeasurement", "No. of Pixel Layers per Track", 10, 0.0, 10.0);
    if (!stripLayerwithMeasurementH_)
      stripLayerwithMeasurementH_ =
          iBook.book1D("stripLayerwithMeasurement", "No. of Strip Layers per Track", 20, 0.0, 20.0);

    if (!beamSpotXYposH_)
      beamSpotXYposH_ = iBook.book1D("beamSpotXYpos", "XY position of beam spot", 40, -4.0, 4.0);
    if (!beamSpotXYposerrH_)
      beamSpotXYposerrH_ = iBook.book1D("beamSpotXYposerr", "Error in XY position of beam spot", 20, 0.0, 4.0);
    if (!beamSpotZposH_)
      beamSpotZposH_ = iBook.book1D("beamSpotZpos", "Z position of beam spot", 100, -20.0, 20.0);
    if (!beamSpotZposerrH_)
      beamSpotZposerrH_ = iBook.book1D("beamSpotZposerr", "Error in Z position of beam spot", 50, 0.0, 5.0);

    if (!vertexXposH_)
      vertexXposH_ = iBook.book1D("vertexXpos", "Vertex X position", 50, -1.0, 1.0);
    if (!vertexYposH_)
      vertexYposH_ = iBook.book1D("vertexYpos", "Vertex Y position", 50, -1.0, 1.0);
    if (!vertexZposH_)
      vertexZposH_ = iBook.book1D("vertexZpos", "Vertex Z position", 100, -20.0, 20.0);
    if (!nVertexH_)
      nVertexH_ = iBook.book1D("nVertex", "# of vertices", 60, -0.5, 59.5);

    if (!nPixBarrelH_)
      nPixBarrelH_ = iBook.book1D("nHitPixelBarrel", "No. of hits in Pixel Barrel per Track", 20, 0, 20.0);
    if (!nPixEndcapH_)
      nPixEndcapH_ = iBook.book1D("nHitPixelEndcap", "No. of hits in Pixel Endcap per Track", 20, 0, 20.0);
    if (!nStripTIBH_)
      nStripTIBH_ = iBook.book1D("nHitStripTIB", "No. of hits in Strip TIB per Track", 30, 0, 30.0);
    if (!nStripTOBH_)
      nStripTOBH_ = iBook.book1D("nHitStripTOB", "No. of hits in Strip TOB per Track", 30, 0, 30.0);
    if (!nStripTECH_)
      nStripTECH_ = iBook.book1D("nHitStripTEC", "No. of hits in Strip TEC per Track", 30, 0, 30.0);
    if (!nStripTIDH_)
      nStripTIDH_ = iBook.book1D("nHitStripTID", "No. of hits in Strip TID per Tracks", 30, 0, 30.0);

    if (!nTracksH_)
      nTracksH_ = iBook.book1D("nTracks", "No. of Tracks", 100, -0.5, 999.5);
  }
  if (isMC_) {
    if (!bunchCrossingH_)
      bunchCrossingH_ = iBook.book1D("bunchCrossing", "Bunch Crosssing", 60, 0, 60.0);
    if (!nPUH_)
      nPUH_ = iBook.book1D("nPU", "No of Pileup", 60, 0, 60.0);
    if (!trueNIntH_)
      trueNIntH_ = iBook.book1D("trueNInt", "True no of Interactions", 60, 0, 60.0);
  }
  // Exclusive histograms
  if (!nHitsVspTH_)
    nHitsVspTH_ = iBook.bookProfile("nHitsVspT",
                                    "Number of Hits Vs pT",
                                    TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                                    TrackPtHistoPar.getParameter<double>("Xmin"),
                                    TrackPtHistoPar.getParameter<double>("Xmax"),
                                    0.0,
                                    0.0,
                                    "g");
  if (!nHitsVsnVtxH_)
    nHitsVsnVtxH_ = iBook.bookProfile("nHitsVsnVtx", "Number of Hits Vs Number of Vertex", 100, 0.0, 50, 0.0, 0.0, "g");
  if (!nHitsVsEtaH_)
    nHitsVsEtaH_ = iBook.bookProfile("nHitsVsEta",
                                     "Number of Hits Vs Eta",
                                     TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                     TrackEtaHistoPar.getParameter<double>("Xmin"),
                                     TrackEtaHistoPar.getParameter<double>("Xmax"),
                                     0.0,
                                     0.0,
                                     "g");
  if (!nHitsVsCosThetaH_)
    nHitsVsCosThetaH_ =
        iBook.bookProfile("nHitsVsCosTheta", "Number of Hits Vs Cos(Theta)", 50, -1.0, 1.0, 0.0, 0.0, "g");
  if (!nHitsVsPhiH_)
    nHitsVsPhiH_ = iBook.bookProfile("nHitsVsPhi", "Number of Hits Vs Phi", 100, -3.5, 3.5, 0.0, 0.0, "g");

  if (!nLostHitsVspTH_)
    nLostHitsVspTH_ = iBook.bookProfile("nLostHitsVspT",
                                        "Number of Lost Hits Vs pT",
                                        TrackPtHistoPar.getParameter<int32_t>("Xbins"),
                                        TrackPtHistoPar.getParameter<double>("Xmin"),
                                        TrackPtHistoPar.getParameter<double>("Xmax"),
                                        0.0,
                                        0.0,
                                        "g");
  if (!nLostHitsVsEtaH_)
    nLostHitsVsEtaH_ = iBook.bookProfile("nLostHitsVsEta",
                                         "Number of Lost Hits Vs Eta",
                                         TrackEtaHistoPar.getParameter<int32_t>("Xbins"),
                                         TrackEtaHistoPar.getParameter<double>("Xmin"),
                                         TrackEtaHistoPar.getParameter<double>("Xmax"),
                                         0.0,
                                         0.0,
                                         "g");
  if (!nLostHitsVsCosThetaH_)
    nLostHitsVsCosThetaH_ =
        iBook.bookProfile("nLostHitsVsCosTheta", "Number of Lost Hits Vs Cos(Theta)", 50, -1.0, 1.0, 0.0, 0.0, "g");
  if (!nLostHitsVsPhiH_)
    nLostHitsVsPhiH_ = iBook.bookProfile("nLostHitsVsPhi", "Number of Lost Hits Vs Phi", 100, -3.5, 3.5, 0.0, 0.0, "g");

  // On and off-track cluster properties
  if (!hOnTrkClusChargeThinH_)
    hOnTrkClusChargeThinH_ =
        iBook.book1D("hOnTrkClusChargeThin", "On-track Cluster Charge (Thin Sensor)", 100, 0, 1000);
  if (!hOnTrkClusWidthThinH_)
    hOnTrkClusWidthThinH_ = iBook.book1D("hOnTrkClusWidthThin", "On-track Cluster Width (Thin Sensor)", 20, -0.5, 19.5);
  if (!hOnTrkClusChargeThickH_)
    hOnTrkClusChargeThickH_ =
        iBook.book1D("hOnTrkClusChargeThick", "On-track Cluster Charge (Thick Sensor)", 100, 0, 1000);
  if (!hOnTrkClusWidthThickH_)
    hOnTrkClusWidthThickH_ =
        iBook.book1D("hOnTrkClusWidthThick", "On-track Cluster Width (Thick Sensor)", 20, -0.5, 19.5);

  if (!hOffTrkClusChargeThinH_)
    hOffTrkClusChargeThinH_ =
        iBook.book1D("hOffTrkClusChargeThin", "Off-track Cluster Charge (Thin Sensor)", 100, 0, 1000);
  if (!hOffTrkClusWidthThinH_)
    hOffTrkClusWidthThinH_ =
        iBook.book1D("hOffTrkClusWidthThin", "Off-track Cluster Width (Thin Sensor)", 20, -0.5, 19.5);
  if (!hOffTrkClusChargeThickH_)
    hOffTrkClusChargeThickH_ =
        iBook.book1D("hOffTrkClusChargeThick", "Off-track Cluster Charge (Thick Sensor)", 100, 0, 1000);
  if (!hOffTrkClusWidthThickH_)
    hOffTrkClusWidthThickH_ =
        iBook.book1D("hOffTrkClusWidthThick", "Off-track Cluster Width (Thick Sensor)", 20, -0.5, 19.5);
}
void StandaloneTrackMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  const TrackerGeometry& tkGeom = (*geomHandle);

  siStripClusterInfo_.initEvent(iSetup);

  // Primary vertex collection
  edm::Handle<reco::VertexCollection> vertexColl;
  iEvent.getByToken(vertexToken_, vertexColl);

  // Beam spot
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  // Track collection
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackToken_, tracks);

  // Access PU information
  double wfac = 1.0;  // for data
  if (!iEvent.isRealData()) {
    edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
    iEvent.getByToken(puSummaryToken_, PupInfo);

    if (verbose_)
      edm::LogInfo("StandaloneTrackMonitor") << "nPUColl = " << PupInfo->size();
    for (auto const& v : *PupInfo) {
      int bx = v.getBunchCrossing();
      if (bunchCrossingH_)
        bunchCrossingH_->Fill(bx);
      if (bx == 0) {
        if (nPUH_)
          nPUH_->Fill(v.getPU_NumInteractions());
        int ntrueInt = v.getTrueNumInteractions();
        if (trueNIntH_)
          trueNIntH_->Fill(ntrueInt);
        if (doPUCorrection_)
          if (ntrueInt > -1 && ntrueInt < int(vpu_.size()))
            wfac = vpu_.at(ntrueInt);
      }
    }
  }
  if (verbose_)
    edm::LogInfo("StandaloneTrackMonitor") << "PU reweight factor = " << wfac;

  if (!vertexColl.isValid())
    edm::LogError("DqmTrackStudy") << "Error! Failed to get reco::Vertex Collection, " << vertexTag_;
  if (haveAllHistograms_) {
    int nvtx = (vertexColl.isValid() ? vertexColl->size() : 0);
    nVertexH_->Fill(nvtx);
  }

  int ntracks = 0;
  if (tracks.isValid()) {
    edm::LogInfo("StandaloneTrackMonitor") << "Total # of Tracks: " << tracks->size();
    if (verbose_)
      edm::LogInfo("StandaloneTrackMonitor") << "Total # of Tracks: " << tracks->size();
    reco::Track::TrackQuality quality = reco::Track::qualityByName(trackQuality_);
    for (auto const& track : *tracks) {
      if (!track.quality(quality))
        continue;

      ++ntracks;

      double eta = track.eta();
      double theta = track.theta();
      double phi = track.phi();
      double pt = track.pt();

      const reco::HitPattern& hitp = track.hitPattern();
      double nValidTrackerHits = hitp.numberOfValidTrackerHits();
      nHitsVsEtaH_->Fill(eta, nValidTrackerHits);
      nHitsVsCosThetaH_->Fill(std::cos(theta), nValidTrackerHits);
      nHitsVsPhiH_->Fill(phi, nValidTrackerHits);
      nHitsVspTH_->Fill(pt, nValidTrackerHits);
      nHitsVsnVtxH_->Fill(vertexColl->size(), nValidTrackerHits);

      int nLostHits = track.numberOfLostHits();
      nLostHitsVspTH_->Fill(pt, nLostHits);
      nLostHitsVsEtaH_->Fill(eta, nLostHits);
      nLostHitsVsCosThetaH_->Fill(std::cos(theta), nLostHits);
      nLostHitsVsPhiH_->Fill(phi, nLostHits);

      double nValidPixelHits = hitp.numberOfValidPixelHits();
      double nValidStripHits = hitp.numberOfValidStripHits();
      double pixelLayersWithMeasurement = hitp.pixelLayersWithMeasurement();
      double stripLayersWithMeasurement = hitp.stripLayersWithMeasurement();

      if (haveAllHistograms_) {
        double etaError = track.etaError();
        double thetaError = track.thetaError();
        double phiError = track.phiError();
        double p = track.p();
        double ptError = track.ptError();
        double qoverp = track.qoverp();
        double qoverpError = track.qoverpError();
        double charge = track.charge();

        double trackerLayersWithMeasurement = hitp.trackerLayersWithMeasurement();

        double dxy = track.dxy(beamSpot->position());
        double dxyError = track.dxyError();
        double dz = track.dz(beamSpot->position());
        double dzError = track.dzError();

        double trkd0 = track.d0();
        double chi2 = track.chi2();
        double ndof = track.ndof();
        double vx = track.vx();
        double vy = track.vy();
        double vz = track.vz();

        // Fill the histograms
        trackEtaH_->Fill(eta, wfac);
        trackEtaerrH_->Fill(etaError, wfac);
        trackCosThetaH_->Fill(std::cos(theta), wfac);
        trackThetaerrH_->Fill(thetaError, wfac);
        trackPhiH_->Fill(phi, wfac);
        trackPhierrH_->Fill(phiError, wfac);
        trackPH_->Fill(p, wfac);
        trackPtH_->Fill(pt, wfac);
        if (pt <= 2)
          trackPtUpto2GeVH_->Fill(pt, wfac);
        if (pt >= 10)
          trackPtOver10GeVH_->Fill(pt, wfac);
        trackPterrH_->Fill(ptError, wfac);
        trackqOverpH_->Fill(qoverp, wfac);
        trackqOverperrH_->Fill(qoverpError, wfac);
        trackChargeH_->Fill(charge, wfac);
        trackChi2H_->Fill(chi2, wfac);
        trackd0H_->Fill(trkd0, wfac);
        tracknDOFH_->Fill(ndof, wfac);
        trackChi2bynDOFH_->Fill(chi2 / ndof, wfac);

        nlostHitsH_->Fill(nLostHits, wfac);
        nvalidTrackerHitsH_->Fill(nValidTrackerHits, wfac);
        nvalidPixelHitsH_->Fill(nValidPixelHits, wfac);
        nvalidStripHitsH_->Fill(nValidStripHits, wfac);

        trkLayerwithMeasurementH_->Fill(trackerLayersWithMeasurement, wfac);
        pixelLayerwithMeasurementH_->Fill(pixelLayersWithMeasurement, wfac);
        stripLayerwithMeasurementH_->Fill(stripLayersWithMeasurement, wfac);

        beamSpotXYposH_->Fill(dxy, wfac);
        beamSpotXYposerrH_->Fill(dxyError, wfac);
        beamSpotZposH_->Fill(dz, wfac);
        beamSpotZposerrH_->Fill(dzError, wfac);

        vertexXposH_->Fill(vx, wfac);
        vertexYposH_->Fill(vy, wfac);
        vertexZposH_->Fill(vz, wfac);
      }
      int nPixBarrel = 0, nPixEndcap = 0, nStripTIB = 0, nStripTOB = 0, nStripTEC = 0, nStripTID = 0;
      for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
        const TrackingRecHit& hit = (**it);
        if (hit.isValid()) {
          if (hit.geographicalId().det() == DetId::Tracker) {
            int subdetId = hit.geographicalId().subdetId();
            if (subdetId == PixelSubdetector::PixelBarrel)
              ++nPixBarrel;
            else if (subdetId == PixelSubdetector::PixelEndcap)
              ++nPixEndcap;
            else if (subdetId == StripSubdetector::TIB)
              ++nStripTIB;
            else if (subdetId == StripSubdetector::TOB)
              ++nStripTOB;
            else if (subdetId == StripSubdetector::TEC)
              ++nStripTEC;
            else if (subdetId == StripSubdetector::TID)
              ++nStripTID;

            // Find on-track clusters
            processHit(hit, iSetup, tkGeom, wfac);
          }
        }
      }
      if (verbose_)
        edm::LogInfo("StandaloneTrackMonitor")
            << " >>> HITs: nPixBarrel: " << nPixBarrel << " nPixEndcap: " << nPixEndcap << " nStripTIB: " << nStripTIB
            << " nStripTOB: " << nStripTOB << " nStripTEC: " << nStripTEC << " nStripTID: " << nStripTID;
      if (haveAllHistograms_) {
        nPixBarrelH_->Fill(nPixBarrel, wfac);
        nPixEndcapH_->Fill(nPixEndcap, wfac);
        nStripTIBH_->Fill(nStripTIB, wfac);
        nStripTOBH_->Fill(nStripTOB, wfac);
        nStripTECH_->Fill(nStripTEC, wfac);
        nStripTIDH_->Fill(nStripTID, wfac);
      }
    }
  } else {
    edm::LogError("DqmTrackStudy") << "Error! Failed to get reco::Track collection, " << trackTag_;
  }
  if (haveAllHistograms_)
    nTracksH_->Fill(ntracks);

  // off track cluster properties
  processClusters(iEvent, iSetup, tkGeom, wfac);
}
void StandaloneTrackMonitor::processClusters(edm::Event const& iEvent,
                                             edm::EventSetup const& iSetup,
                                             const TrackerGeometry& tkGeom,
                                             double wfac) {
  // SiStripClusters
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterHandle;
  iEvent.getByToken(clusterToken_, clusterHandle);

  if (clusterHandle.isValid()) {
    // Loop on Dets
    for (edmNew::DetSetVector<SiStripCluster>::const_iterator dsvit = clusterHandle->begin();
         dsvit != clusterHandle->end();
         ++dsvit) {
      uint32_t detId = dsvit->id();
      std::map<uint32_t, std::set<const SiStripCluster*> >::iterator jt = clusterMap_.find(detId);
      bool detid_found = (jt != clusterMap_.end()) ? true : false;

      // Loop on Clusters
      for (edmNew::DetSet<SiStripCluster>::const_iterator clusit = dsvit->begin(); clusit != dsvit->end(); ++clusit) {
        if (detid_found) {
          std::set<const SiStripCluster*>& s = jt->second;
          if (s.find(&*clusit) != s.end())
            continue;
        }

        siStripClusterInfo_.setCluster(*clusit, detId);
        float charge = siStripClusterInfo_.charge();
        float width = siStripClusterInfo_.width();

        const GeomDetUnit* detUnit = tkGeom.idToDetUnit(detId);
        float thickness = detUnit->surface().bounds().thickness();  // unit cm
        if (thickness > 0.035) {
          hOffTrkClusChargeThickH_->Fill(charge, wfac);
          hOffTrkClusWidthThickH_->Fill(width, wfac);
        } else {
          hOffTrkClusChargeThinH_->Fill(charge, wfac);
          hOffTrkClusWidthThinH_->Fill(width, wfac);
        }
      }
    }
  } else {
    edm::LogError("StandaloneTrackMonitor") << "ClusterCollection " << clusterTag_ << " not valid!!" << std::endl;
  }
}
void StandaloneTrackMonitor::processHit(const TrackingRecHit& recHit,
                                        edm::EventSetup const& iSetup,
                                        const TrackerGeometry& tkGeom,
                                        double wfac) {
  uint32_t detid = recHit.geographicalId();
  const GeomDetUnit* detUnit = tkGeom.idToDetUnit(detid);
  float thickness = detUnit->surface().bounds().thickness();  // unit cm

  auto const& thit = static_cast<BaseTrackerRecHit const&>(recHit);
  if (!thit.isValid())
    return;

  auto const& clus = thit.firstClusterRef();
  if (!clus.isValid())
    return;
  if (!clus.isStrip())
    return;

  if (thit.isMatched()) {
    const SiStripMatchedRecHit2D& matchedHit = dynamic_cast<const SiStripMatchedRecHit2D&>(recHit);

    auto& clusterM = matchedHit.monoCluster();
    siStripClusterInfo_.setCluster(clusterM, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(siStripClusterInfo_.width(), wfac);
    } else {
      hOnTrkClusChargeThinH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(siStripClusterInfo_.width(), wfac);
    }
    addClusterToMap(detid, &clusterM);

    auto& clusterS = matchedHit.stereoCluster();
    siStripClusterInfo_.setCluster(clusterS, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(siStripClusterInfo_.width(), wfac);
    } else {
      hOnTrkClusChargeThinH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(siStripClusterInfo_.width(), wfac);
    }
    addClusterToMap(detid, &clusterS);
  } else {
    auto& cluster = clus.stripCluster();
    siStripClusterInfo_.setCluster(cluster, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(siStripClusterInfo_.width(), wfac);
    } else {
      hOnTrkClusChargeThinH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(siStripClusterInfo_.width(), wfac);
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
  } else {
    std::set<const SiStripCluster*>& s = it->second;
    s.insert(cluster);
  }
}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StandaloneTrackMonitor);

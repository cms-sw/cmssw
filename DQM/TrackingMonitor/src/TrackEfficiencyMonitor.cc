/*
 *  See header file for a description of this class.
 *
 *  \author Jeremy Andrea
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackEfficiencyMonitor.h"
#include <string>

// needed to compute the efficiency

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include <RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h>

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

//-----------------------------------------------------------------------------------
TrackEfficiencyMonitor::TrackEfficiencyMonitor(const edm::ParameterSet& iConfig)
//-----------------------------------------------------------------------------------
{
  dqmStore_ = edm::Service<DQMStore>().operator->();

  theRadius_ = iConfig.getParameter<double>("theRadius");
  theMaxZ_ = iConfig.getParameter<double>("theMaxZ");
  isBFieldOff_ = iConfig.getParameter<bool>("isBFieldOff");
  trackEfficiency_ = iConfig.getParameter<bool>("trackEfficiency");
  theTKTracksLabel_ = iConfig.getParameter<edm::InputTag>("TKTrackCollection");
  theSTATracksLabel_ = iConfig.getParameter<edm::InputTag>("STATrackCollection");
  muonToken_ = consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag>("muoncoll"));

  theTKTracksToken_ = consumes<reco::TrackCollection>(theTKTracksLabel_);
  theSTATracksToken_ = consumes<reco::TrackCollection>(theSTATracksLabel_);

  conf_ = iConfig;
}

//-----------------------------------------------------------------------------------
TrackEfficiencyMonitor::~TrackEfficiencyMonitor()
//-----------------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------------
void TrackEfficiencyMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& /* iRun */,
                                            edm::EventSetup const& /* iSetup */)
//-----------------------------------------------------------------------------------
{
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName");
  std::string AlgoName = conf_.getParameter<std::string>("AlgoName");

  ibooker.setCurrentFolder(MEFolderName);

  //
  int muonXBin = conf_.getParameter<int>("muonXBin");
  double muonXMin = conf_.getParameter<double>("muonXMin");
  double muonXMax = conf_.getParameter<double>("muonXMax");

  histname = "muonX_";
  muonX = ibooker.book1D(histname + AlgoName, histname + AlgoName, muonXBin, muonXMin, muonXMax);
  muonX->setAxisTitle("");

  //
  int muonYBin = conf_.getParameter<int>("muonYBin");
  double muonYMin = conf_.getParameter<double>("muonYMin");
  double muonYMax = conf_.getParameter<double>("muonYMax");

  histname = "muonY_";
  muonY = ibooker.book1D(histname + AlgoName, histname + AlgoName, muonYBin, muonYMin, muonYMax);
  muonY->setAxisTitle("");

  //
  int muonZBin = conf_.getParameter<int>("muonZBin");
  double muonZMin = conf_.getParameter<double>("muonZMin");
  double muonZMax = conf_.getParameter<double>("muonZMax");

  histname = "muonZ_";
  muonZ = ibooker.book1D(histname + AlgoName, histname + AlgoName, muonZBin, muonZMin, muonZMax);
  muonZ->setAxisTitle("");

  //
  int muonEtaBin = conf_.getParameter<int>("muonEtaBin");
  double muonEtaMin = conf_.getParameter<double>("muonEtaMin");
  double muonEtaMax = conf_.getParameter<double>("muonEtaMax");

  histname = "muonEta_";
  muonEta = ibooker.book1D(histname + AlgoName, histname + AlgoName, muonEtaBin, muonEtaMin, muonEtaMax);
  muonEta->setAxisTitle("");

  //
  int muonPhiBin = conf_.getParameter<int>("muonPhiBin");
  double muonPhiMin = conf_.getParameter<double>("muonPhiMin");
  double muonPhiMax = conf_.getParameter<double>("muonPhiMax");

  histname = "muonPhi_";
  muonPhi = ibooker.book1D(histname + AlgoName, histname + AlgoName, muonPhiBin, muonPhiMin, muonPhiMax);
  muonPhi->setAxisTitle("");

  //
  int muonD0Bin = conf_.getParameter<int>("muonD0Bin");
  double muonD0Min = conf_.getParameter<double>("muonD0Min");
  double muonD0Max = conf_.getParameter<double>("muonD0Max");

  histname = "muonD0_";
  muonD0 = ibooker.book1D(histname + AlgoName, histname + AlgoName, muonD0Bin, muonD0Min, muonD0Max);
  muonD0->setAxisTitle("");

  //
  int muonCompatibleLayersBin = conf_.getParameter<int>("muonCompatibleLayersBin");
  double muonCompatibleLayersMin = conf_.getParameter<double>("muonCompatibleLayersMin");
  double muonCompatibleLayersMax = conf_.getParameter<double>("muonCompatibleLayersMax");

  histname = "muonCompatibleLayers_";
  muonCompatibleLayers = ibooker.book1D(histname + AlgoName,
                                        histname + AlgoName,
                                        muonCompatibleLayersBin,
                                        muonCompatibleLayersMin,
                                        muonCompatibleLayersMax);
  muonCompatibleLayers->setAxisTitle("");

  //------------------------------------------------------------------------------------

  //
  int trackXBin = conf_.getParameter<int>("trackXBin");
  double trackXMin = conf_.getParameter<double>("trackXMin");
  double trackXMax = conf_.getParameter<double>("trackXMax");

  histname = "trackX_";
  trackX = ibooker.book1D(histname + AlgoName, histname + AlgoName, trackXBin, trackXMin, trackXMax);
  trackX->setAxisTitle("");

  //
  int trackYBin = conf_.getParameter<int>("trackYBin");
  double trackYMin = conf_.getParameter<double>("trackYMin");
  double trackYMax = conf_.getParameter<double>("trackYMax");

  histname = "trackY_";
  trackY = ibooker.book1D(histname + AlgoName, histname + AlgoName, trackYBin, trackYMin, trackYMax);
  trackY->setAxisTitle("");

  //
  int trackZBin = conf_.getParameter<int>("trackZBin");
  double trackZMin = conf_.getParameter<double>("trackZMin");
  double trackZMax = conf_.getParameter<double>("trackZMax");

  histname = "trackZ_";
  trackZ = ibooker.book1D(histname + AlgoName, histname + AlgoName, trackZBin, trackZMin, trackZMax);
  trackZ->setAxisTitle("");

  //
  int trackEtaBin = conf_.getParameter<int>("trackEtaBin");
  double trackEtaMin = conf_.getParameter<double>("trackEtaMin");
  double trackEtaMax = conf_.getParameter<double>("trackEtaMax");

  histname = "trackEta_";
  trackEta = ibooker.book1D(histname + AlgoName, histname + AlgoName, trackEtaBin, trackEtaMin, trackEtaMax);
  trackEta->setAxisTitle("");

  //
  int trackPhiBin = conf_.getParameter<int>("trackPhiBin");
  double trackPhiMin = conf_.getParameter<double>("trackPhiMin");
  double trackPhiMax = conf_.getParameter<double>("trackPhiMax");

  histname = "trackPhi_";
  trackPhi = ibooker.book1D(histname + AlgoName, histname + AlgoName, trackPhiBin, trackPhiMin, trackPhiMax);
  trackPhi->setAxisTitle("");

  //
  int trackD0Bin = conf_.getParameter<int>("trackD0Bin");
  double trackD0Min = conf_.getParameter<double>("trackD0Min");
  double trackD0Max = conf_.getParameter<double>("trackD0Max");

  histname = "trackD0_";
  trackD0 = ibooker.book1D(histname + AlgoName, histname + AlgoName, trackD0Bin, trackD0Min, trackD0Max);
  trackD0->setAxisTitle("");

  //
  int trackCompatibleLayersBin = conf_.getParameter<int>("trackCompatibleLayersBin");
  double trackCompatibleLayersMin = conf_.getParameter<double>("trackCompatibleLayersMin");
  double trackCompatibleLayersMax = conf_.getParameter<double>("trackCompatibleLayersMax");

  histname = "trackCompatibleLayers_";
  trackCompatibleLayers = ibooker.book1D(histname + AlgoName,
                                         histname + AlgoName,
                                         trackCompatibleLayersBin,
                                         trackCompatibleLayersMin,
                                         trackCompatibleLayersMax);
  trackCompatibleLayers->setAxisTitle("");

  //------------------------------------------------------------------------------------

  //
  int deltaXBin = conf_.getParameter<int>("deltaXBin");
  double deltaXMin = conf_.getParameter<double>("deltaXMin");
  double deltaXMax = conf_.getParameter<double>("deltaXMax");

  histname = "deltaX_";
  deltaX = ibooker.book1D(histname + AlgoName, histname + AlgoName, deltaXBin, deltaXMin, deltaXMax);
  deltaX->setAxisTitle("");

  //
  int deltaYBin = conf_.getParameter<int>("deltaYBin");
  double deltaYMin = conf_.getParameter<double>("deltaYMin");
  double deltaYMax = conf_.getParameter<double>("deltaYMax");

  histname = "deltaY_";
  deltaY = ibooker.book1D(histname + AlgoName, histname + AlgoName, deltaYBin, deltaYMin, deltaYMax);
  deltaY->setAxisTitle("");

  //
  int signDeltaXBin = conf_.getParameter<int>("signDeltaXBin");
  double signDeltaXMin = conf_.getParameter<double>("signDeltaXMin");
  double signDeltaXMax = conf_.getParameter<double>("signDeltaXMax");

  histname = "signDeltaX_";
  signDeltaX = ibooker.book1D(histname + AlgoName, histname + AlgoName, signDeltaXBin, signDeltaXMin, signDeltaXMax);
  signDeltaX->setAxisTitle("");

  //
  int signDeltaYBin = conf_.getParameter<int>("signDeltaYBin");
  double signDeltaYMin = conf_.getParameter<double>("signDeltaYMin");
  double signDeltaYMax = conf_.getParameter<double>("signDeltaYMax");

  histname = "signDeltaY_";
  signDeltaY = ibooker.book1D(histname + AlgoName, histname + AlgoName, signDeltaYBin, signDeltaYMin, signDeltaYMax);
  signDeltaY->setAxisTitle("");

  histname = "GlobalMuonPtEtaPhi_LowPt_";
  GlobalMuonPtEtaPhiLowPt = ibooker.book2D(histname + AlgoName, histname + AlgoName, 20, -2.4, 2.4, 20, -3.25, 3.25);
  GlobalMuonPtEtaPhiLowPt->setAxisTitle("");

  histname = "StandaloneMuonPtEtaPhi_LowPt_";
  StandaloneMuonPtEtaPhiLowPt =
      ibooker.book2D(histname + AlgoName, histname + AlgoName, 20, -2.4, 2.4, 20, -3.25, 3.25);
  StandaloneMuonPtEtaPhiLowPt->setAxisTitle("");

  histname = "GlobalMuonPtEtaPhi_HighPt_";
  GlobalMuonPtEtaPhiHighPt = ibooker.book2D(histname + AlgoName, histname + AlgoName, 20, -2.4, 2.4, 20, -3.25, 3.25);
  GlobalMuonPtEtaPhiHighPt->setAxisTitle("");

  histname = "StandaloneMuonPtEtaPhi_HighPt_";
  StandaloneMuonPtEtaPhiHighPt =
      ibooker.book2D(histname + AlgoName, histname + AlgoName, 20, -2.4, 2.4, 20, -3.25, 3.25);
  StandaloneMuonPtEtaPhiHighPt->setAxisTitle("");
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyMonitor::beginJob(void)
//-----------------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------------
void TrackEfficiencyMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
//-----------------------------------------------------------------------------------
{
  edm::Handle<reco::TrackCollection> tkTracks;
  iEvent.getByToken(theTKTracksToken_, tkTracks);
  edm::Handle<reco::TrackCollection> staTracks;
  iEvent.getByToken(theSTATracksToken_, staTracks);
  edm::ESHandle<NavigationSchool> nav;
  iSetup.get<NavigationSchoolRecord>().get("CosmicNavigationSchool", nav);
  iSetup.get<CkfComponentsRecord>().get(measurementTrackerHandle);

  //initialize values
  failedToPropagate = 0;
  nCompatibleLayers = 0;
  findDetLayer = false;

  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theTTrackBuilder);
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", thePropagator);
  iSetup.get<IdealMagneticFieldRecord>().get(bField);
  iSetup.get<TrackerRecoGeometryRecord>().get(theTracker);
  theNavigation = new DirectTrackerNavigation(theTracker);

  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByToken(muonToken_, muons);
  if (!muons.isValid())
    return;
  for (edm::View<reco::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
    if ((*muon).pt() < 5)
      continue;
    if (fabs((*muon).eta()) > 2.4)
      continue;
    if ((*muon).vertexNormalizedChi2() > 10)
      continue;
    if ((*muon).isStandAloneMuon() and (*muon).isGlobalMuon()) {
      if ((*muon).pt() < 20)
        GlobalMuonPtEtaPhiLowPt->Fill((*muon).eta(), (*muon).phi());
      else
        GlobalMuonPtEtaPhiHighPt->Fill((*muon).eta(), (*muon).phi());
    }
    if ((*muon).isStandAloneMuon()) {
      if ((*muon).pt() < 20)
        StandaloneMuonPtEtaPhiLowPt->Fill((*muon).eta(), (*muon).phi());
      else
        StandaloneMuonPtEtaPhiHighPt->Fill((*muon).eta(), (*muon).phi());
    }
  }
  if (trackEfficiency_) {
    //---------------------------------------------------
    // Select muons with good quality
    // If B field is on, no up-down matching between the muons
    //---------------------------------------------------
    bool isGoodMuon = false;
    double mudd0 = 0., mudphi = 0., muddsz = 0., mudeta = 0.;
    if (isBFieldOff_) {
      if (staTracks->size() == 2) {
        for (unsigned int bindex = 0; bindex < staTracks->size(); ++bindex) {
          if (0 == bindex) {
            mudd0 += (*staTracks)[bindex].d0();
            mudphi += (*staTracks)[bindex].phi();
            muddsz += (*staTracks)[bindex].dsz();
            mudeta += (*staTracks)[bindex].eta();
          }
          if (1 == bindex) {
            mudd0 -= (*staTracks)[bindex].d0();
            mudphi -= (*staTracks)[bindex].phi();
            muddsz -= (*staTracks)[bindex].dsz();
            mudeta -= (*staTracks)[bindex].eta();
          }
        }
        if ((fabs(mudd0) < 15.0) && (fabs(mudphi) < 0.045) && (fabs(muddsz) < 20.0) && (fabs(mudeta) < 0.060))
          isGoodMuon = true;
      }

      if (isGoodMuon)
        testTrackerTracks(tkTracks, staTracks, *nav.product());

    } else if (staTracks->size() == 1 || staTracks->size() == 2)
      testTrackerTracks(tkTracks, staTracks, *nav.product());
  }

  if (!trackEfficiency_ && tkTracks->size() == 1) {
    if ((tkTracks->front()).normalizedChi2() < 5 && (tkTracks->front()).hitPattern().numberOfValidHits() > 8)
      testSTATracks(tkTracks, staTracks);
  }

  delete theNavigation;
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyMonitor::endJob(void)
//-----------------------------------------------------------------------------------
{
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if (outputMEsInRootFile) {
    dqmStore_->save(outputFileName);
  }

  //if ( theNavigation ) delete theNavigation;
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyMonitor::testTrackerTracks(edm::Handle<reco::TrackCollection> tkTracks,
                                               edm::Handle<reco::TrackCollection> staTracks,
                                               const NavigationSchool& navigationSchool)
//-----------------------------------------------------------------------------------
{
  const std::string metname = "testStandAloneMuonTracks";

  //---------------------------------------------------
  // get the index of the "up" muon
  // histograms will only be computed for the "up" muon
  //---------------------------------------------------

  int nUpMuon = 0;
  int idxUpMuon = -1;
  for (unsigned int i = 0; i < staTracks->size(); i++) {
    if (checkSemiCylinder((*staTracks)[i]) == TrackEfficiencyMonitor::Up) {
      nUpMuon++;
      idxUpMuon = i;
    }
  }

  if (nUpMuon == 1) {
    //---------------------------------------------------
    //get the muon informations
    //---------------------------------------------------

    reco::TransientTrack staTT = theTTrackBuilder->build((*staTracks)[idxUpMuon]);
    double ipX = staTT.impactPointState().globalPosition().x();
    double ipY = staTT.impactPointState().globalPosition().y();
    double ipZ = staTT.impactPointState().globalPosition().z();
    double eta = staTT.impactPointState().globalDirection().eta();
    double phi = staTT.impactPointState().globalDirection().phi();
    double d0 = (*staTracks)[idxUpMuon].d0();

    TrajectoryStateOnSurface theTSOS = staTT.outermostMeasurementState();
    TrajectoryStateOnSurface theTSOSCompLayers = staTT.outermostMeasurementState();

    //---------------------------------------------------
    //check if the muon is in the tracker acceptance
    //---------------------------------------------------
    bool isInTrackerAcceptance = false;
    isInTrackerAcceptance = trackerAcceptance(theTSOS, theRadius_, theMaxZ_);

    //---------------------------------------------------st
    //count the number of compatible layers
    //---------------------------------------------------
    nCompatibleLayers = compatibleLayers(navigationSchool, theTSOSCompLayers);

    if (isInTrackerAcceptance && (*staTracks)[idxUpMuon].hitPattern().numberOfValidHits() > 28) {
      //---------------------------------------------------
      //count the number of good muon candidates
      //---------------------------------------------------

      TrajectoryStateOnSurface staState;
      LocalVector diffLocal;

      bool isTrack = false;
      if (!tkTracks->empty()) {
        //---------------------------------------------------
        //look for associated tracks
        //---------------------------------------------------
        float DR2min = 1000;
        reco::TrackCollection::const_iterator closestTrk = tkTracks->end();

        for (reco::TrackCollection::const_iterator tkTrack = tkTracks->begin(); tkTrack != tkTracks->end(); ++tkTrack) {
          reco::TransientTrack tkTT = theTTrackBuilder->build(*tkTrack);
          TrajectoryStateOnSurface tkInner = tkTT.innermostMeasurementState();
          staState = thePropagator->propagate(staTT.outermostMeasurementState(), tkInner.surface());
          failedToPropagate = 1;

          if (staState.isValid()) {
            failedToPropagate = 0;
            diffLocal = tkInner.localPosition() - staState.localPosition();
            double DR2 = diffLocal.x() * diffLocal.x() + diffLocal.y() * diffLocal.y();
            if (DR2 < DR2min) {
              DR2min = DR2;
              closestTrk = tkTrack;
            }
            if (pow(DR2, 0.5) < 100.)
              isTrack = true;
          }
        }

        if (DR2min != 1000) {
          reco::TransientTrack tkTT = theTTrackBuilder->build(*closestTrk);
          TrajectoryStateOnSurface tkInner = tkTT.innermostMeasurementState();
          staState = thePropagator->propagate(staTT.outermostMeasurementState(), tkInner.surface());
          deltaX->Fill(diffLocal.x());
          deltaY->Fill(diffLocal.y());
          signDeltaX->Fill(diffLocal.x() /
                           (tkInner.localError().positionError().xx() + staState.localError().positionError().xx()));
          signDeltaY->Fill(diffLocal.y() /
                           (tkInner.localError().positionError().yy() + staState.localError().positionError().yy()));
        }
      }

      if (failedToPropagate == 0) {
        muonX->Fill(ipX);
        muonY->Fill(ipY);
        muonZ->Fill(ipZ);
        muonEta->Fill(eta);
        muonPhi->Fill(phi);
        muonD0->Fill(d0);
        muonCompatibleLayers->Fill(nCompatibleLayers);

        if (isTrack) {
          trackX->Fill(ipX);
          trackY->Fill(ipY);
          trackZ->Fill(ipZ);
          trackEta->Fill(eta);
          trackPhi->Fill(phi);
          trackD0->Fill(d0);
          trackCompatibleLayers->Fill(nCompatibleLayers);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyMonitor::testSTATracks(edm::Handle<reco::TrackCollection> tkTracks,
                                           edm::Handle<reco::TrackCollection> staTracks)
//-----------------------------------------------------------------------------------
{
  reco::TransientTrack tkTT = theTTrackBuilder->build(tkTracks->front());
  double ipX = tkTT.impactPointState().globalPosition().x();
  double ipY = tkTT.impactPointState().globalPosition().y();
  double ipZ = tkTT.impactPointState().globalPosition().z();
  double eta = tkTT.impactPointState().globalDirection().eta();
  double phi = tkTT.impactPointState().globalDirection().phi();
  double d0 = (*tkTracks)[0].d0();

  TrajectoryStateOnSurface tkInner = tkTT.innermostMeasurementState();
  LocalVector diffLocal;
  TrajectoryStateOnSurface staState;
  bool isTrack = false;

  if (!staTracks->empty()) {
    //---------------------------------------------------
    //look for associated muons
    //---------------------------------------------------

    float DR2min = 1000;
    reco::TrackCollection::const_iterator closestTrk = staTracks->end();
    //----------------------loop on tracker tracks:
    for (reco::TrackCollection::const_iterator staTrack = staTracks->begin(); staTrack != staTracks->end();
         ++staTrack) {
      if (checkSemiCylinder(*staTrack) == TrackEfficiencyMonitor::Up) {
        reco::TransientTrack staTT = theTTrackBuilder->build(*staTrack);
        failedToPropagate = 1;
        staState = thePropagator->propagate(staTT.outermostMeasurementState(), tkInner.surface());

        if (staState.isValid()) {
          failedToPropagate = 0;
          diffLocal = tkInner.localPosition() - staState.localPosition();

          double DR2 = diffLocal.x() * diffLocal.x() + diffLocal.y() * diffLocal.y();
          if (DR2 < DR2min) {
            DR2min = DR2;
            closestTrk = staTrack;
          }
          if (pow(DR2, 0.5) < 100.)
            isTrack = true;
        }
      }
    }
  }

  if (failedToPropagate == 0) {
    trackX->Fill(ipX);
    trackY->Fill(ipY);
    trackZ->Fill(ipZ);
    trackEta->Fill(eta);
    trackPhi->Fill(phi);
    trackD0->Fill(d0);

    if (isTrack) {
      muonX->Fill(ipX);
      muonY->Fill(ipY);
      muonZ->Fill(ipZ);
      muonEta->Fill(eta);
      muonPhi->Fill(phi);
      muonD0->Fill(d0);
    }
  }
}

//-----------------------------------------------------------------------------------
TrackEfficiencyMonitor::SemiCylinder TrackEfficiencyMonitor::checkSemiCylinder(const reco::Track& tk)
//-----------------------------------------------------------------------------------
{
  return tk.innerPosition().phi() > 0 ? TrackEfficiencyMonitor::Up : TrackEfficiencyMonitor::Down;
}

//-----------------------------------------------------------------------------------
bool TrackEfficiencyMonitor::trackerAcceptance(TrajectoryStateOnSurface theTSOS, double theRadius, double theMaxZ)
//-----------------------------------------------------------------------------------
{
  //---------------------------------------------------
  // check if the muon is in the tracker acceptance
  // check the compatibility with a cylinder of radius "theRadius"
  //---------------------------------------------------

  //Propagator*  theTmpPropagator = new SteppingHelixPropagator(&*bField,anyDirection);

  //Propagator*  theTmpPropagator = &*thePropagator;
  Propagator* theTmpPropagator = &*thePropagator->clone();

  if (theTSOS.globalPosition().y() < 0)
    theTmpPropagator->setPropagationDirection(oppositeToMomentum);
  else
    theTmpPropagator->setPropagationDirection(alongMomentum);

  Cylinder::PositionType pos0;
  Cylinder::RotationType rot0;
  const Cylinder::CylinderPointer cyl = Cylinder::build(pos0, rot0, theRadius);
  TrajectoryStateOnSurface tsosAtCyl = theTmpPropagator->propagate(*theTSOS.freeState(), *cyl);
  double accept = false;
  if (tsosAtCyl.isValid()) {
    if (fabs(tsosAtCyl.globalPosition().z()) < theMaxZ) {
      Cylinder::PositionType pos02;
      Cylinder::RotationType rot02;
      const Cylinder::CylinderPointer cyl2 = Cylinder::build(pos02, rot02, theRadius - 10);
      TrajectoryStateOnSurface tsosAtCyl2 = theTmpPropagator->propagate(*tsosAtCyl.freeState(), *cyl2);
      if (tsosAtCyl2.isValid()) {
        Cylinder::PositionType pos03;
        Cylinder::RotationType rot03;
        const Cylinder::CylinderPointer cyl3 = Cylinder::build(pos03, rot03, theRadius);
        TrajectoryStateOnSurface tsosAtCyl3 = theTmpPropagator->propagate(*tsosAtCyl2.freeState(), *cyl3);
        if (tsosAtCyl3.isValid()) {
          accept = true;
        }
      }
    }
  }
  delete theTmpPropagator;
  //muon propagated to the barrel cylinder
  return accept;
}

//-----------------------------------------------------------------------------------
int TrackEfficiencyMonitor::compatibleLayers(const NavigationSchool& navigationSchool, TrajectoryStateOnSurface theTSOS)
//-----------------------------------------------------------------------------------
{
  //---------------------------------------------------
  // check the number of compatible layers
  //---------------------------------------------------

  std::vector<const BarrelDetLayer*> barrelTOBLayers = measurementTrackerHandle->geometricSearchTracker()->tobLayers();

  unsigned int layers = 0;
  for (unsigned int k = 0; k < barrelTOBLayers.size(); k++) {
    const DetLayer* firstLay = barrelTOBLayers[barrelTOBLayers.size() - 1 - k];

    //Propagator*  theTmpPropagator = new SteppingHelixPropagator(&*bField,anyDirection);

    Propagator* theTmpPropagator = &*thePropagator->clone();
    theTmpPropagator->setPropagationDirection(alongMomentum);

    TrajectoryStateOnSurface startTSOS = theTmpPropagator->propagate(*theTSOS.freeState(), firstLay->surface());

    std::vector<const DetLayer*> trackCompatibleLayers;

    findDetLayer = true;
    bool isUpMuon = false;
    bool firstdtep = true;

    if (startTSOS.isValid()) {
      if (firstdtep)
        layers++;

      int nwhile = 0;

      //for other compatible layers
      while (startTSOS.isValid() && firstLay && findDetLayer) {
        if (firstdtep && startTSOS.globalPosition().y() > 0)
          isUpMuon = true;

        if (firstdtep) {
          std::vector<const DetLayer*> firstCompatibleLayers;
          firstCompatibleLayers.push_back(firstLay);
          std::pair<TrajectoryStateOnSurface, const DetLayer*> nextLayer =
              findNextLayer(theTSOS, firstCompatibleLayers, isUpMuon);
          firstdtep = false;
        } else {
          trackCompatibleLayers = navigationSchool.nextLayers(*firstLay, *(startTSOS.freeState()), alongMomentum);
          if (!trackCompatibleLayers.empty()) {
            std::pair<TrajectoryStateOnSurface, const DetLayer*> nextLayer =
                findNextLayer(startTSOS, trackCompatibleLayers, isUpMuon);
            if (firstLay != nextLayer.second) {
              firstLay = nextLayer.second;
              startTSOS = nextLayer.first;
              layers++;
            } else
              firstLay = nullptr;
          }
        }
        nwhile++;
        if (nwhile > 100)
          break;
      }
      delete theTmpPropagator;
      break;
    }
    delete theTmpPropagator;
  }
  return layers;
}

//-----------------------------------------------------------------------------------
std::pair<TrajectoryStateOnSurface, const DetLayer*> TrackEfficiencyMonitor::findNextLayer(
    TrajectoryStateOnSurface sTSOS, const std::vector<const DetLayer*>& trackCompatibleLayers, bool isUpMuon)
//-----------------------------------------------------------------------------------
{
  //Propagator*  theTmpPropagator = new SteppingHelixPropagator(&*bField,anyDirection);

  Propagator* theTmpPropagator = &*thePropagator->clone();
  theTmpPropagator->setPropagationDirection(alongMomentum);

  std::vector<const DetLayer*>::const_iterator itl;
  findDetLayer = false;
  for (itl = trackCompatibleLayers.begin(); itl != trackCompatibleLayers.end(); ++itl) {
    TrajectoryStateOnSurface tsos = theTmpPropagator->propagate(*(sTSOS.freeState()), (**itl).surface());
    if (tsos.isValid()) {
      sTSOS = tsos;
      findDetLayer = true;

      break;
    }
  }
  std::pair<TrajectoryStateOnSurface, const DetLayer*> blabla;
  blabla.first = sTSOS;
  blabla.second = &**itl;
  delete theTmpPropagator;
  return blabla;
}

DEFINE_FWK_MODULE(TrackEfficiencyMonitor);

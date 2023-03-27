////////////////////////////////////////////////////////////////////////////////
// Package:          CalibTracker/SiStripHitResolution
// Class:            HitResol
// Original Authors: Denis Gele and Kathryn Coldham (adapted from HitEff)
//                   modified by Khawla Jaffel for CPE studies
//                   ported to cmssw by M. Musich
//
///////////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>
#include <string>
#include <iostream>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEfficiencyHelpers.h"
#include "CalibTracker/SiStripHitResolution/interface/HitResol.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

// ROOT includes
#include "TMath.h"
#include "TH1F.h"

//
// constructors and destructor
//
using namespace std;
HitResol::HitResol(const edm::ParameterSet& conf)
    : scalerToken_(consumes<LumiScalersCollection>(conf.getParameter<edm::InputTag>("lumiScalers"))),
      combinatorialTracks_token_(
          consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("combinatorialTracks"))),
      tjToken_(consumes<std::vector<Trajectory> >(conf.getParameter<edm::InputTag>("trajectories"))),
      topoToken_(esConsumes()),
      geomToken_(esConsumes()),
      cpeToken_(esConsumes(edm::ESInputTag("", "StripCPEfromTrackAngle"))),
      siStripQualityToken_(esConsumes()),
      magFieldToken_(esConsumes()),
      addLumi_(conf.getUntrackedParameter<bool>("addLumi", false)),
      DEBUG_(conf.getParameter<bool>("Debug")),
      cutOnTracks_(conf.getUntrackedParameter<bool>("cutOnTracks", false)),
      momentumCut_(conf.getUntrackedParameter<double>("MomentumCut", 3.)),
      compSettings_(conf.getUntrackedParameter<int>("CompressionSettings", -1)),
      usePairsOnly_(conf.getUntrackedParameter<unsigned int>("UsePairsOnly", 1)),
      layers_(conf.getParameter<int>("Layer")),
      trackMultiplicityCut_(conf.getUntrackedParameter<unsigned int>("trackMultiplicity", 100)) {
  usesResource(TFileService::kSharedResource);
}

void HitResol::beginJob() {
  edm::Service<TFileService> fs;
  if (compSettings_ > 0) {
    edm::LogInfo("SiStripHitResolution:HitResol") << "the compressions settings are:" << compSettings_ << std::endl;
    fs->file().SetCompressionSettings(compSettings_);
  }

  reso = fs->make<TTree>("reso", "tree hit pairs for resolution studies");
  reso->Branch("momentum", &mymom, "momentum/F");
  reso->Branch("numHits", &numHits, "numHits/I");
  reso->Branch("trackChi2", &ProbTrackChi2, "trackChi2/F");
  reso->Branch("detID1", &iidd1, "detID1/I");
  reso->Branch("pitch1", &mypitch1, "pitch1/F");
  reso->Branch("clusterW1", &clusterWidth, "clusterW1/I");
  reso->Branch("expectedW1", &expWidth, "expectedW1/F");
  reso->Branch("atEdge1", &atEdge, "atEdge1/F");
  reso->Branch("simpleRes", &simpleRes, "simpleRes/F");
  reso->Branch("detID2", &iidd2, "detID2/I");
  reso->Branch("clusterW2", &clusterWidth_2, "clusterW2/I");
  reso->Branch("expectedW2", &expWidth_2, "expectedW2/F");
  reso->Branch("atEdge2", &atEdge_2, "atEdge2/F");
  reso->Branch("pairPath", &pairPath, "pairPath/F");
  reso->Branch("hitDX", &hitDX, "hitDX/F");
  reso->Branch("trackDX", &trackDX, "trackDX/F");
  reso->Branch("trackDXE", &trackDXE, "trackDXE/F");
  reso->Branch("trackParamX", &trackParamX, "trackParamX/F");
  reso->Branch("trackParamY", &trackParamY, "trackParamY/F");
  reso->Branch("trackParamDXDZ", &trackParamDXDZ, "trackParamDXDZ/F");
  reso->Branch("trackParamDYDZ", &trackParamDYDZ, "trackParamDYDZ/F");
  reso->Branch("trackParamXE", &trackParamXE, "trackParamXE/F");
  reso->Branch("trackParamYE", &trackParamYE, "trackParamYE/F");
  reso->Branch("trackParamDXDZE", &trackParamDXDZE, "trackParamDXDZE/F");
  reso->Branch("trackParamDYDZE", &trackParamDYDZE, "trackParamDYDZE/F");
  reso->Branch("pairsOnly", &pairsOnly, "pairsOnly/I");
  treso = fs->make<TTree>("treso", "tree tracks  for resolution studies");
  treso->Branch("track_momentum", &track_momentum, "track_momentum/F");
  treso->Branch("track_pt", &track_pt, "track_pt/F");
  treso->Branch("track_eta", &track_eta, "track_eta/F");
  treso->Branch("track_phi", &track_phi, "track_phi/F");
  treso->Branch("track_trackChi2", &track_trackChi2, "track_trackChi2/F");
  treso->Branch("track_width", &expWidth, "track_width/F");  // from 1D HIT
  treso->Branch("NumberOf_tracks", &NumberOf_tracks, "NumberOf_tracks/I");

  events = 0;
  EventTrackCKF = 0;

  histos2d_["track_phi_vs_eta"] = new TH2F("track_phi_vs_eta", ";track phi;track eta", 60, -3.5, 3.5, 60, -3., 3.);
  histos2d_["residual_vs_trackMomentum"] = new TH2F("residual_vs_trackMomentum",
                                                    ";track momentum [GeV]; x_{pred_track} - x_{reco_hit} [#mum]",
                                                    60,
                                                    0.,
                                                    10.,
                                                    60,
                                                    0.,
                                                    200.);
  histos2d_["residual_vs_trackPt"] = new TH2F(
      "residual_vs_trackPt", ";track p_{T}[GeV];x_{pred_track} - x_{reco_hit} [#mum]", 60, 0., 10., 60, 0., 200.);
  histos2d_["residual_vs_trackEta"] =
      new TH2F("residual_vs_trackEta", ";track #eta;x_{pred_track} - x_{reco_hit} [#mum]", 60, 0., 3., 60, 0., 200.);
  histos2d_["residual_vs_trackPhi"] =
      new TH2F("residual_vs_trackPhi", ";track #phi;x_{pred_track} - x_{reco_hit} [#mum]", 60, 0., 3.5, 60, 0., 200.);
  histos2d_["residual_vs_expectedWidth"] = new TH2F(
      "residual_vs_expectedWidth", ";track Width;x_{pred_track} - x_{reco_hit} [#mum]", 3, 0., 3., 60, 0., 200.);
  histos2d_["numHits_vs_residual"] =
      new TH2F("numHits_vs_residual", ";x_{pred_track} - x_{reco_hit} [#mum];N Hits", 60, 0., 200., 15, 0., 15.);
}

void HitResol::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &es.getData(topoToken_);

  LogDebug("SiStripHitResolution:HitResol") << "beginning analyze from HitResol" << endl;

  using namespace edm;
  using namespace reco;

  // Step A: Get Inputs

  int run_nr = e.id().run();
  int ev_nr = e.id().event();

  // get the tracks
  edm::Handle<reco::TrackCollection> trackCollectionCKF;
  e.getByToken(combinatorialTracks_token_, trackCollectionCKF);
  const reco::TrackCollection* tracksCKF = trackCollectionCKF.product();

  // get the trajectory collection
  edm::Handle<std::vector<Trajectory> > trajectoryCollectionHandle;
  e.getByToken(tjToken_, trajectoryCollectionHandle);
  const TrajectoryCollection* trajectoryCollection = trajectoryCollectionHandle.product();

  //get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker = es.getHandle(geomToken_);
  const TrackerGeometry* tkgeom = &(*tracker);

  //get Cluster Parameter Estimator
  edm::ESHandle<StripClusterParameterEstimator> parameterestimator = es.getHandle(cpeToken_);
  const StripClusterParameterEstimator& stripcpe(*parameterestimator);

  // get the SiStripQuality records
  edm::ESHandle<SiStripQuality> SiStripQuality_ = es.getHandle(siStripQualityToken_);

  // get the magnetic field
  const MagneticField* magField_ = &es.getData(magFieldToken_);

  events++;

  // List of variables for SiStripHitResolution ntuple
  mymom = 0;
  numHits = 0;
  ProbTrackChi2 = 0;
  iidd1 = 0;
  mypitch1 = 0;
  clusterWidth = 0;
  expWidth = 0;
  atEdge = 0;
  simpleRes = 0;
  iidd2 = 0;
  clusterWidth_2 = 0;
  expWidth_2 = 0;
  atEdge_2 = 0;
  pairPath = 0;
  hitDX = 0;
  trackDX = 0;
  trackDXE = 0;
  trackParamX = 0;
  trackParamY = 0;
  trackParamDXDZ = 0;
  trackParamDYDZ = 0;
  trackParamXE = 0;
  trackParamYE = 0;
  trackParamDXDZE = 0;
  trackParamDYDZE = 0;
  pairsOnly = 0;

  LogDebug("HitResol") << "Starting analysis, nrun nevent, tracksCKF->size(): " << run_nr << " " << ev_nr << " "
                       << tracksCKF->size() << std::endl;

  for (unsigned int iT = 0; iT < tracksCKF->size(); ++iT) {
    track_momentum = tracksCKF->at(iT).p();
    track_pt = tracksCKF->at(iT).p();
    track_eta = tracksCKF->at(iT).eta();
    track_phi = tracksCKF->at(iT).phi();
    track_trackChi2 = ChiSquaredProbability((double)(tracksCKF->at(iT).chi2()), (double)(tracksCKF->at(iT).ndof()));
    treso->Fill();
  }

  histos2d_["track_phi_vs_eta"]->Fill(track_phi, track_eta);

  // loop over trajectories from refit
  for (const auto& traj : *trajectoryCollection) {
    const auto& TMeas = traj.measurements();
    // Loop on each measurement and take it into consideration
    //--------------------------------------------------------
    for (auto itm = TMeas.cbegin(); itm != TMeas.cend(); ++itm) {
      if (!itm->updatedState().isValid()) {
        LogDebug("HitResol") << "trajectory measurement not valid" << std::endl;
        continue;
      }

      const TransientTrackingRecHit::ConstRecHitPointer mypointhit = itm->recHit();
      const TrackingRecHit* myhit = (*itm->recHit()).hit();

      ProbTrackChi2 = 0;
      numHits = 0;

      LogDebug("HitResol") << "TrackChi2 =  "
                           << ChiSquaredProbability((double)(traj.chiSquared()), (double)(traj.ndof(false))) << "\n"
                           << "itm->updatedState().globalMomentum().perp(): "
                           << itm->updatedState().globalMomentum().perp() << "\n"
                           << "numhits " << traj.foundHits() << std::endl;

      numHits = traj.foundHits();
      ProbTrackChi2 = ChiSquaredProbability((double)(traj.chiSquared()), (double)(traj.ndof(false)));

      mymom = itm->updatedState().globalMomentum().perp();

      //Now for the first hit
      TrajectoryStateOnSurface mytsos = itm->updatedState();
      const auto hit1 = itm->recHit();
      DetId id1 = hit1->geographicalId();
      if (id1.subdetId() < StripSubdetector::TIB || id1.subdetId() > StripSubdetector::TEC)
        continue;

      if (hit1->isValid() && mymom > momentumCut_ &&
          (id1.subdetId() >= StripSubdetector::TIB && id1.subdetId() <= StripSubdetector::TEC)) {
        const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(hit1->geographicalId()));
        const StripTopology& Topo = stripdet->specificTopology();
        int Nstrips = Topo.nstrips();
        mypitch1 = stripdet->surface().bounds().width() / Topo.nstrips();

        const auto det = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(mypointhit->geographicalId()));

        TrajectoryStateOnSurface mytsos = itm->updatedState();
        LocalVector trackDirection = mytsos.localDirection();
        LocalVector drift = stripcpe.driftDirection(stripdet);

        const auto hit1d = dynamic_cast<const SiStripRecHit1D*>(myhit);

        if (hit1d) {
          getSimHitRes(det, trackDirection, *hit1d, expWidth, &mypitch1, drift);
          clusterWidth = hit1d->cluster()->amplitudes().size();
          uint16_t firstStrip = hit1d->cluster()->firstStrip();
          uint16_t lastStrip = firstStrip + (hit1d->cluster()->amplitudes()).size() - 1;
          atEdge = (firstStrip == 0 || lastStrip == (Nstrips - 1));
        }

        const auto hit2d = dynamic_cast<const SiStripRecHit2D*>(myhit);

        if (hit2d) {
          getSimHitRes(det, trackDirection, *hit2d, expWidth, &mypitch1, drift);
          clusterWidth = hit2d->cluster()->amplitudes().size();
          uint16_t firstStrip = hit2d->cluster()->firstStrip();
          uint16_t lastStrip = firstStrip + (hit2d->cluster()->amplitudes()).size() - 1;
          atEdge = (firstStrip == 0 || lastStrip == (Nstrips - 1));
        }

        simpleRes =
            getSimpleRes(&(*itm));  // simple resolution by using the track re-fit forward and backward predicted state

        histos2d_["residual_vs_trackMomentum"]->Fill(itm->updatedState().globalMomentum().mag(),
                                                     simpleRes * 10000);   // reso in cm *10000 == micro-meter
        histos2d_["residual_vs_trackPt"]->Fill(mymom, simpleRes * 10000);  // reso in cm *10000 == micro-meter
        histos2d_["residual_vs_trackEta"]->Fill(itm->updatedState().globalMomentum().eta(), simpleRes * 10000);
        histos2d_["residual_vs_trackPhi"]->Fill(itm->updatedState().globalMomentum().phi(), simpleRes * 10000);
        histos2d_["residual_vs_expectedWidth"]->Fill(expWidth, simpleRes * 10000);
        histos2d_["numHits_vs_residual"]->Fill(simpleRes * 10000, numHits);

        // Now to see if there is a match - pair method - hit in overlapping sensors
        vector<TrajectoryMeasurement>::const_iterator itTraj2 = TMeas.end();  // last hit along the fitted track

        for (auto itmCompare = itm - 1;
             // start to compare from the 5th hit
             itmCompare >= TMeas.cbegin() && itmCompare > itm - 4;
             --itmCompare) {
          const auto hit2 = itmCompare->recHit();
          if (!hit2->isValid())
            continue;
          DetId id2 = hit2->geographicalId();

          //must be from the same detector and layer
          iidd1 = hit1->geographicalId().rawId();
          iidd2 = hit2->geographicalId().rawId();
          if (id1.subdetId() != id2.subdetId() || ::checkLayer(iidd1, tTopo) != ::checkLayer(iidd2, tTopo))
            break;
          //must both be stereo if one is
          if (tTopo->isStereo(id1) != tTopo->isStereo(id2))
            continue;
          //A check i dont completely understand but might as well keep there
          if (tTopo->glued(id1) == id1.rawId())
            LogDebug("HitResol") << "BAD GLUED: Have glued layer with id = " << id1.rawId()
                                 << " and glued id = " << tTopo->glued(id1) << "  and stereo = " << tTopo->isStereo(id1)
                                 << endl;
          if (tTopo->glued(id2) == id2.rawId())
            LogDebug("HitResol") << "BAD GLUED: Have glued layer with id = " << id2.rawId()
                                 << " and glued id = " << tTopo->glued(id2) << "  and stereo = " << tTopo->isStereo(id2)
                                 << endl;

          itTraj2 = itmCompare;
          break;
        }

        if (itTraj2 == TMeas.cend()) {
        } else {
          LogDebug("HitResol") << "Found overlapping sensors " << std::endl;
          pairsOnly = usePairsOnly_;

          //We found one....let's fill in the truth info!
          TrajectoryStateOnSurface tsos_2 = itTraj2->updatedState();
          LocalVector trackDirection_2 = tsos_2.localDirection();
          const auto myhit2 = itTraj2->recHit();
          const auto myhit_2 = (*itTraj2->recHit()).hit();
          const auto stripdet_2 = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(myhit2->geographicalId()));
          const StripTopology& Topo_2 = stripdet_2->specificTopology();
          int Nstrips_2 = Topo_2.nstrips();
          float mypitch_2 = stripdet_2->surface().bounds().width() / Topo_2.nstrips();

          if (mypitch1 != mypitch_2)
            return;  // for PairsOnly

          const auto det_2 = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(myhit2->geographicalId()));

          LocalVector drift_2 = stripcpe.driftDirection(stripdet_2);

          const auto hit1d_2 = dynamic_cast<const SiStripRecHit1D*>(myhit_2);
          if (hit1d_2) {
            getSimHitRes(det_2, trackDirection_2, *hit1d_2, expWidth_2, &mypitch_2, drift_2);
            clusterWidth_2 = hit1d_2->cluster()->amplitudes().size();
            uint16_t firstStrip_2 = hit1d_2->cluster()->firstStrip();
            uint16_t lastStrip_2 = firstStrip_2 + (hit1d_2->cluster()->amplitudes()).size() - 1;
            atEdge_2 = (firstStrip_2 == 0 || lastStrip_2 == (Nstrips_2 - 1));
          }

          const auto hit2d_2 = dynamic_cast<const SiStripRecHit2D*>(myhit_2);
          if (hit2d_2) {
            getSimHitRes(det_2, trackDirection_2, *hit2d_2, expWidth_2, &mypitch_2, drift_2);
            clusterWidth_2 = hit2d_2->cluster()->amplitudes().size();
            uint16_t firstStrip_2 = hit2d_2->cluster()->firstStrip();
            uint16_t lastStrip_2 = firstStrip_2 + (hit2d_2->cluster()->amplitudes()).size() - 1;
            atEdge_2 = (firstStrip_2 == 0 || lastStrip_2 == (Nstrips_2 - 1));
          }

          // if(pairsOnly && (pitch != pitch2) ) fill = false;

          // Make AnalyticalPropagator to use in getPairParameters
          AnalyticalPropagator mypropagator(magField_, anyDirection);

          if (!getPairParameters(&(*magField_),
                                 mypropagator,
                                 &(*itTraj2),
                                 &(*itm),
                                 pairPath,
                                 hitDX,
                                 trackDX,
                                 trackDXE,
                                 trackParamX,
                                 trackParamY,
                                 trackParamDXDZ,
                                 trackParamDYDZ,
                                 trackParamXE,
                                 trackParamYE,
                                 trackParamDXDZE,
                                 trackParamDYDZE)) {
          } else {
            LogDebug("HitResol") << " \n\n\n"
                                 << " momentum       " << mymom << "\n"
                                 << " numHits        " << numHits << "\n"
                                 << " trackChi2      " << ProbTrackChi2 << "\n"
                                 << " detID1         " << iidd1 << "\n"
                                 << " pitch1         " << mypitch1 << "\n"
                                 << " clusterW1      " << clusterWidth << "\n"
                                 << " expectedW1     " << expWidth << "\n"
                                 << " atEdge1        " << atEdge << "\n"
                                 << " simpleRes      " << simpleRes << "\n"
                                 << " detID2         " << iidd2 << "\n"
                                 << " clusterW2      " << clusterWidth_2 << "\n"
                                 << " expectedW2     " << expWidth_2 << "\n"
                                 << " atEdge2        " << atEdge_2 << "\n"
                                 << " pairPath       " << pairPath << "\n"
                                 << " hitDX          " << hitDX << "\n"
                                 << " trackDX        " << trackDX << "\n"
                                 << " trackDXE       " << trackDXE << "\n"
                                 << " trackParamX	   " << trackParamX << "\n"
                                 << " trackParamY	   " << trackParamY << "\n"
                                 << " trackParamDXDZ " << trackParamDXDZ << "\n"
                                 << " trackParamDYDZ " << trackParamDYDZ << "\n"
                                 << " trackParamXE   " << trackParamXE << "\n"
                                 << " trackParamYE   " << trackParamYE << "\n"
                                 << " trackParamDXDZE" << trackParamDXDZE << "\n"
                                 << " trackParamDYDZE" << trackParamDYDZE << std::endl;
            reso->Fill();
          }
        }  //itTraj2 != TMeas.end()
      }    //hit1->isValid()....
    }      // itm
  }        // it
}

void HitResol::endJob() {
  LogDebug("SiStripHitResolution:HitResol") << " Events Analysed             " << events << endl;
  LogDebug("SiStripHitResolution:HitResol") << " Number Of Tracked events    " << EventTrackCKF << endl;

  reso->GetDirectory()->cd();
  reso->Write();
  treso->Write();
}

double HitResol::checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters,
                                  double xx,
                                  double xerr) {
  double error = sqrt(parameters.second.xx() + xerr * xerr);
  double separation = abs(parameters.first.x() - xx);
  double consistency = separation / error;
  return consistency;
}

void HitResol::getSimHitRes(const GeomDetUnit* det,
                            const LocalVector& trackdirection,
                            const TrackingRecHit& recHit,
                            float& trackWidth,
                            float* pitch,
                            LocalVector& drift) {
  const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(det);
  const auto& topol = dynamic_cast<const StripTopology&>(stripdet->topology());

  LocalPoint position = recHit.localPosition();
  (*pitch) = topol.localPitch(position);

  float anglealpha = 0;
  if (trackdirection.z() != 0) {
    anglealpha = atan(trackdirection.x() / trackdirection.z()) * TMath::RadToDeg();
  }

  //  LocalVector drift = stripcpe.driftDirection(stripdet);
  float thickness = stripdet->surface().bounds().thickness();
  float tanalpha = tan(anglealpha * TMath::DegToRad());
  float tanalphaL = drift.x() / drift.z();
  (trackWidth) = fabs((thickness / (*pitch)) * tanalpha - (thickness / (*pitch)) * tanalphaL);
}

double HitResol::getSimpleRes(const TrajectoryMeasurement* traj1) {
  TrajectoryStateOnSurface theCombinedPredictedState;

  if (traj1->backwardPredictedState().isValid())
    theCombinedPredictedState =
        TrajectoryStateCombiner().combine(traj1->forwardPredictedState(), traj1->backwardPredictedState());
  else
    theCombinedPredictedState = traj1->forwardPredictedState();

  if (!theCombinedPredictedState.isValid()) {
    return -100;
  }

  const TransientTrackingRecHit::ConstRecHitPointer& firstRecHit = traj1->recHit();
  double recHitX_1 = firstRecHit->localPosition().x();
  return (theCombinedPredictedState.localPosition().x() - recHitX_1);
}

//traj1 is the matched trajectory...traj2 is the original
bool HitResol::getPairParameters(const MagneticField* magField_,
                                 AnalyticalPropagator& propagator,
                                 const TrajectoryMeasurement* traj1,
                                 const TrajectoryMeasurement* traj2,
                                 float& pairPath,
                                 float& hitDX,
                                 float& trackDX,
                                 float& trackDXE,
                                 float& trackParamX,
                                 float& trackParamY,
                                 float& trackParamDXDZ,
                                 float& trackParamDYDZ,
                                 float& trackParamXE,
                                 float& trackParamYE,
                                 float& trackParamDXDZE,
                                 float& trackParamDYDZE) {
  pairPath = 0;
  hitDX = 0;
  trackDX = 0;
  trackDXE = 0;

  trackParamX = 0;
  trackParamY = 0;
  trackParamDXDZ = 0;
  trackParamDYDZ = 0;
  trackParamXE = 0;
  trackParamYE = 0;
  trackParamDXDZE = 0;
  trackParamDYDZE = 0;

  TrajectoryStateCombiner combiner_;

  // backward predicted state at module 1
  const TrajectoryStateOnSurface& bwdPred1 = traj1->backwardPredictedState();
  if (!bwdPred1.isValid())
    return false;
  LogDebug("HitResol") << "momentum from backward predicted state = " << bwdPred1.globalMomentum().mag() << endl;
  // forward predicted state at module 2
  const TrajectoryStateOnSurface& fwdPred2 = traj2->forwardPredictedState();
  LogDebug("HitResol") << "momentum from forward predicted state = " << fwdPred2.globalMomentum().mag() << endl;
  if (!fwdPred2.isValid())
    return false;
  // extrapolate fwdPred2 to module 1
  TrajectoryStateOnSurface fwdPred2At1 = propagator.propagate(fwdPred2, bwdPred1.surface());
  if (!fwdPred2At1.isValid())
    return false;
  // combine fwdPred2At1 with bwdPred1 (ref. state, best estimate without hits 1 and 2)
  TrajectoryStateOnSurface comb1 = combiner_.combine(bwdPred1, fwdPred2At1);
  if (!comb1.isValid())
    return false;

  //
  // propagation of reference parameters to module 2
  //
  std::pair<TrajectoryStateOnSurface, double> tsosWithS = propagator.propagateWithPath(comb1, fwdPred2.surface());
  TrajectoryStateOnSurface comb1At2 = tsosWithS.first;
  if (!comb1At2.isValid())
    return false;
  //distance of propagation from one surface to the next==could cut here
  pairPath = tsosWithS.second;
  if (TMath::Abs(pairPath) > 15)
    return false;  //cut to remove hit pairs > 15 cm apart

  // local parameters and errors on module 1
  AlgebraicVector5 pars = comb1.localParameters().vector();
  AlgebraicSymMatrix55 errs = comb1.localError().matrix();
  //number 3 is predX
  double predX1 = pars[3];
  //track fitted parameters in local coordinates for position 0
  (trackParamX) = pars[3];
  (trackParamY) = pars[4];
  (trackParamDXDZ) = pars[1];
  (trackParamDYDZ) = pars[2];
  (trackParamXE) = TMath::Sqrt(errs(3, 3));
  (trackParamYE) = TMath::Sqrt(errs(4, 4));
  (trackParamDXDZE) = TMath::Sqrt(errs(1, 1));
  (trackParamDYDZE) = TMath::Sqrt(errs(2, 2));

  // local parameters and errors on module 2
  pars = comb1At2.localParameters().vector();
  errs = comb1At2.localError().matrix();
  double predX2 = pars[3];

  ////
  //// jacobians (local-to-global@1,global 1-2,global-to-local@2)
  ////
  JacobianLocalToCurvilinear jacLocToCurv(comb1.surface(), comb1.localParameters(), *magField_);
  AnalyticalCurvilinearJacobian jacCurvToCurv(
      comb1.globalParameters(), comb1At2.globalPosition(), comb1At2.globalMomentum(), tsosWithS.second);
  JacobianCurvilinearToLocal jacCurvToLoc(comb1At2.surface(), comb1At2.localParameters(), *magField_);
  // combined jacobian local-1-to-local-2
  AlgebraicMatrix55 jacobian = jacLocToCurv.jacobian() * jacCurvToCurv.jacobian() * jacCurvToLoc.jacobian();
  // covariance on module 1
  AlgebraicSymMatrix55 covComb1 = comb1.localError().matrix();
  // variance and correlations for predicted local_x on modules 1 and 2
  double c00 = covComb1(3, 3);
  double c10(0.);
  double c11(0.);
  for (int i = 1; i < 5; ++i) {
    c10 += jacobian(3, i) * covComb1(i, 3);
    for (int j = 1; j < 5; ++j)
      c11 += jacobian(3, i) * covComb1(i, j) * jacobian(3, j);
  }
  // choose relative sign in order to minimize error on difference
  double diff = c00 - 2 * fabs(c10) + c11;
  diff = diff > 0 ? sqrt(diff) : -sqrt(-diff);
  (trackDXE) = diff;
  double relativeXSign_ = c10 > 0 ? -1 : 1;

  (trackDX) = predX1 + relativeXSign_ * predX2;

  double recHitX_1 = traj1->recHit()->localPosition().x();
  double recHitX_2 = traj2->recHit()->localPosition().x();

  (hitDX) = recHitX_1 + relativeXSign_ * recHitX_2;

  return true;
}

void HitResol::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("lumiScalers", edm::InputTag("scalersRawToDigi"));
  desc.add<edm::InputTag>("combinatorialTracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("trajectories", edm::InputTag("generalTracks"));
  desc.addUntracked<int>("CompressionSettings", -1);
  desc.add<int>("Layer", 0);
  desc.add<bool>("Debug", false);
  desc.addUntracked<bool>("addLumi", false);
  desc.addUntracked<bool>("cutOnTracks", false);
  desc.addUntracked<unsigned int>("trackMultiplicity", 100);
  desc.addUntracked<double>("MomentumCut", 3.);
  desc.addUntracked<unsigned int>("UsePairsOnly", 1);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HitResol);

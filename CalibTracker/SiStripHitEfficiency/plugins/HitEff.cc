////////////////////////////////////////////////////////////////////////////////
// Package:          CalibTracker/SiStripHitEfficiency
// Class:            HitEff
// Original Author:  Keith Ulmer--University of Colorado
//                   keith.ulmer@colorado.edu
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
#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"
#include "CalibTracker/SiStripHitEfficiency/plugins/HitEff.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
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
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

// ROOT includes
#include "TMath.h"
#include "TH1F.h"

// custom made printout
#define LOGPRINT edm::LogPrint("SiStripHitEfficiency:HitEff")

//
// constructors and destructor
//

using namespace std;
HitEff::HitEff(const edm::ParameterSet& conf)
    : scalerToken_(consumes<LumiScalersCollection>(conf.getParameter<edm::InputTag>("lumiScalers"))),
      metaDataToken_(consumes<OnlineLuminosityRecord>(conf.getParameter<edm::InputTag>("metadata"))),
      commonModeToken_(mayConsume<edm::DetSetVector<SiStripRawDigi> >(conf.getParameter<edm::InputTag>("commonMode"))),
      siStripClusterInfo_(consumesCollector()),
      combinatorialTracks_token_(
          consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("combinatorialTracks"))),
      trajectories_token_(consumes<std::vector<Trajectory> >(conf.getParameter<edm::InputTag>("trajectories"))),
      trajTrackAsso_token_(consumes<TrajTrackAssociationCollection>(conf.getParameter<edm::InputTag>("trajectories"))),
      clusters_token_(
          consumes<edmNew::DetSetVector<SiStripCluster> >(conf.getParameter<edm::InputTag>("siStripClusters"))),
      digisCol_token_(consumes(conf.getParameter<edm::InputTag>("siStripDigis"))),
      digisVec_token_(consumes(conf.getParameter<edm::InputTag>("siStripDigis"))),
      trackerEvent_token_(consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent"))),
      topoToken_(esConsumes()),
      geomToken_(esConsumes()),
      cpeToken_(esConsumes(edm::ESInputTag("", "StripCPEfromTrackAngle"))),
      siStripQualityToken_(esConsumes()),
      magFieldToken_(esConsumes()),
      measurementTkToken_(esConsumes()),
      chi2MeasurementEstimatorToken_(esConsumes(edm::ESInputTag("", "Chi2"))),
      propagatorToken_(esConsumes(edm::ESInputTag("", "PropagatorWithMaterial"))),
      conf_(conf) {
  usesResource(TFileService::kSharedResource);
  compSettings = conf_.getUntrackedParameter<int>("CompressionSettings", -1);
  layers = conf_.getParameter<int>("Layer");
  DEBUG = conf_.getParameter<bool>("Debug");
  addLumi_ = conf_.getUntrackedParameter<bool>("addLumi", false);
  addCommonMode_ = conf_.getUntrackedParameter<bool>("addCommonMode", false);
  cutOnTracks_ = conf_.getUntrackedParameter<bool>("cutOnTracks", false);
  trackMultiplicityCut_ = conf.getUntrackedParameter<unsigned int>("trackMultiplicity", 100);
  useFirstMeas_ = conf_.getUntrackedParameter<bool>("useFirstMeas", false);
  useLastMeas_ = conf_.getUntrackedParameter<bool>("useLastMeas", false);
  useAllHitsFromTracksWithMissingHits_ =
      conf_.getUntrackedParameter<bool>("useAllHitsFromTracksWithMissingHits", false);
  doMissingHitsRecovery_ = conf_.getUntrackedParameter<bool>("doMissingHitsRecovery", false);

  hitRecoveryCounters.resize(k_END_OF_LAYERS, 0);
  hitTotalCounters.resize(k_END_OF_LAYERS, 0);
}

void HitEff::beginJob() {
  edm::Service<TFileService> fs;
  if (compSettings > 0) {
    edm::LogInfo("SiStripHitEfficiency:HitEff") << "the compressions settings are:" << compSettings << std::endl;
    fs->file().SetCompressionSettings(compSettings);
  }

  traj = fs->make<TTree>("traj", "tree of trajectory positions");
#ifdef ExtendedCALIBTree
  traj->Branch("timeDT", &timeDT, "timeDT/F");
  traj->Branch("timeDTErr", &timeDTErr, "timeDTErr/F");
  traj->Branch("timeDTDOF", &timeDTDOF, "timeDTDOF/I");
  traj->Branch("timeECAL", &timeECAL, "timeECAL/F");
  traj->Branch("dedx", &dedx, "dedx/F");
  traj->Branch("dedxNOM", &dedxNOM, "dedxNOM/I");
  traj->Branch("nLostHits", &nLostHits, "nLostHits/I");
  traj->Branch("chi2", &chi2, "chi2/F");
  traj->Branch("p", &p, "p/F");
#endif
  traj->Branch("TrajGlbX", &TrajGlbX, "TrajGlbX/F");
  traj->Branch("TrajGlbY", &TrajGlbY, "TrajGlbY/F");
  traj->Branch("TrajGlbZ", &TrajGlbZ, "TrajGlbZ/F");
  traj->Branch("TrajLocX", &TrajLocX, "TrajLocX/F");
  traj->Branch("TrajLocY", &TrajLocY, "TrajLocY/F");
  traj->Branch("TrajLocAngleX", &TrajLocAngleX, "TrajLocAngleX/F");
  traj->Branch("TrajLocAngleY", &TrajLocAngleY, "TrajLocAngleY/F");
  traj->Branch("TrajLocErrX", &TrajLocErrX, "TrajLocErrX/F");
  traj->Branch("TrajLocErrY", &TrajLocErrY, "TrajLocErrY/F");
  traj->Branch("ClusterLocX", &ClusterLocX, "ClusterLocX/F");
  traj->Branch("ClusterLocY", &ClusterLocY, "ClusterLocY/F");
  traj->Branch("ClusterLocErrX", &ClusterLocErrX, "ClusterLocErrX/F");
  traj->Branch("ClusterLocErrY", &ClusterLocErrY, "ClusterLocErrY/F");
  traj->Branch("ClusterStoN", &ClusterStoN, "ClusterStoN/F");
  traj->Branch("ResX", &ResX, "ResX/F");
  traj->Branch("ResXSig", &ResXSig, "ResXSig/F");
  traj->Branch("ModIsBad", &ModIsBad, "ModIsBad/i");
  traj->Branch("SiStripQualBad", &SiStripQualBad, "SiStripQualBad/i");
  traj->Branch("withinAcceptance", &withinAcceptance, "withinAcceptance/O");
  traj->Branch("nHits", &nHits, "nHits/I");
  traj->Branch("pT", &pT, "pT/F");
  traj->Branch("highPurity", &highPurity, "highPurity/O");
  traj->Branch("trajHitValid", &trajHitValid, "trajHitValid/i");
  traj->Branch("Id", &Id, "Id/i");
  traj->Branch("run", &run, "run/i");
  traj->Branch("event", &event, "event/i");
  traj->Branch("layer", &whatlayer, "layer/i");
  traj->Branch("tquality", &tquality, "tquality/I");
  traj->Branch("bunchx", &bunchx, "bunchx/I");
  if (addLumi_) {
    traj->Branch("instLumi", &instLumi, "instLumi/F");
    traj->Branch("PU", &PU, "PU/F");
  }
  if (addCommonMode_)
    traj->Branch("commonMode", &commonMode, "commonMode/F");

  events = 0;
  EventTrackCKF = 0;

  totalNbHits = 0;
  missHitPerLayer.resize(k_END_OF_LAYERS, 0);
}

void HitEff::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &es.getData(topoToken_);
  siStripClusterInfo_.initEvent(es);

  //  bool DEBUG = false;

  LogDebug("SiStripHitEfficiency:HitEff") << "beginning analyze from HitEff" << endl;

  using namespace edm;
  using namespace reco;
  // Step A: Get Inputs

  int run_nr = e.id().run();
  int ev_nr = e.id().event();
  int bunch_nr = e.bunchCrossing();

  // Luminosity informations
  edm::Handle<LumiScalersCollection> lumiScalers = e.getHandle(scalerToken_);
  edm::Handle<OnlineLuminosityRecord> metaData = e.getHandle(metaDataToken_);

  instLumi = 0;
  PU = 0;
  if (addLumi_) {
    if (lumiScalers.isValid() && !lumiScalers->empty()) {
      if (lumiScalers->begin() != lumiScalers->end()) {
        instLumi = lumiScalers->begin()->instantLumi();
        PU = lumiScalers->begin()->pileup();
      }
    } else if (metaData.isValid()) {
      instLumi = metaData->instLumi();
      PU = metaData->avgPileUp();
    } else {
      edm::LogWarning("SiStripHitEfficiencyWorker") << "could not find a source for the Luminosity and PU";
    }
  }

  // CM
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > commonModeDigis;
  if (addCommonMode_)
    e.getByToken(commonModeToken_, commonModeDigis);

  //CombinatoriaTrack
  edm::Handle<reco::TrackCollection> trackCollectionCKF;
  //edm::InputTag TkTagCKF = conf_.getParameter<edm::InputTag>("combinatorialTracks");
  e.getByToken(combinatorialTracks_token_, trackCollectionCKF);

  edm::Handle<std::vector<Trajectory> > TrajectoryCollectionCKF;
  //edm::InputTag TkTrajCKF = conf_.getParameter<edm::InputTag>("trajectories");
  e.getByToken(trajectories_token_, TrajectoryCollectionCKF);

  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  e.getByToken(trajTrackAsso_token_, trajTrackAssociationHandle);

  // Clusters
  // get the SiStripClusters from the event
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > theClusters;
  //e.getByLabel("siStripClusters", theClusters);
  e.getByToken(clusters_token_, theClusters);

  //get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker = es.getHandle(geomToken_);
  const TrackerGeometry* tkgeom = &(*tracker);

  //get Cluster Parameter Estimator
  //std::string cpe = conf_.getParameter<std::string>("StripCPE");
  edm::ESHandle<StripClusterParameterEstimator> parameterestimator = es.getHandle(cpeToken_);
  const StripClusterParameterEstimator& stripcpe(*parameterestimator);

  // get the SiStripQuality records
  edm::ESHandle<SiStripQuality> SiStripQuality_ = es.getHandle(siStripQualityToken_);

  const MagneticField* magField_ = &es.getData(magFieldToken_);

  // get the list of module IDs with FED-detected errors
  //  - In Aug-2023, the data format was changed from DetIdCollection to DetIdVector.
  //  - To provide some level of backward-compatibility,
  //    the plugin checks for both types giving preference to the new format.
  //  - If only the old format is available, the collection is
  //    converted to the new format, then used downstream.
  auto const& fedErrorIdsCol_h = e.getHandle(digisCol_token_);
  auto const& fedErrorIdsVec_h = e.getHandle(digisVec_token_);
  if (not fedErrorIdsCol_h.isValid() and not fedErrorIdsVec_h.isValid()) {
    throw cms::Exception("InvalidProductSiStripDetIdsWithFEDErrors")
        << "no valid product for SiStrip DetIds with FED errors (see parameter \"siStripDigis\"), "
           "neither for new format (DetIdVector) nor old format (DetIdCollection)";
  }
  auto const& fedErrorIds = fedErrorIdsVec_h.isValid() ? *fedErrorIdsVec_h : fedErrorIdsCol_h->as_vector();

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle = es.getHandle(measurementTkToken_);

  edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
  //e.getByLabel("MeasurementTrackerEvent", measurementTrackerEvent);
  e.getByToken(trackerEvent_token_, measurementTrackerEvent);

  const MeasurementEstimator* estimator = &es.getData(chi2MeasurementEstimatorToken_);
  const Propagator* thePropagator = &es.getData(propagatorToken_);

  events++;

  // *************** SiStripCluster Collection
  const edmNew::DetSetVector<SiStripCluster>& input = *theClusters;

  //go through clusters to write out global position of good clusters for the layer understudy for comparison
  // Loop through clusters just to print out locations
  // Commented out to avoid discussion, should really be deleted.
  /*
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter = input.begin(); DSViter != input.end(); DSViter++) {
    // DSViter is a vector of SiStripClusters located on a single module
    unsigned int ClusterId = DSViter->id();
    DetId ClusterDetId(ClusterId);
    const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tkgeom->idToDetUnit(ClusterDetId);
    
    edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
    for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter) {
      //iter is a single SiStripCluster
      StripClusterParameterEstimator::LocalValues parameters=stripcpe.localParameters(*iter,*stripdet);
      
      const Surface* surface;
      surface = &(tracker->idToDet(ClusterDetId)->surface());
      LocalPoint lp = parameters.first;
      GlobalPoint gp = surface->toGlobal(lp);
      unsigned int layer = ::checkLayer(ClusterId, tTopo);
            if(DEBUG) LOGPRINT << "Found hit in cluster collection layer = " << layer << " with id = " << ClusterId << "   local X position = " << lp.x() << " +- " << sqrt(parameters.second.xx()) << "   matched/stereo/rphi = " << ((ClusterId & 0x3)==0) << "/" << ((ClusterId & 0x3)==1) << "/" << ((ClusterId & 0x3)==2) << endl;
    }
  }
  */

  // Tracking
  const reco::TrackCollection* tracksCKF = trackCollectionCKF.product();
  LogDebug("SiStripHitEfficiency:HitEff") << "number ckf tracks found = " << tracksCKF->size() << endl;
  //if (tracksCKF->size() == 1 ){
  if (!tracksCKF->empty()) {
    if (cutOnTracks_ && (tracksCKF->size() >= trackMultiplicityCut_))
      return;
    if (cutOnTracks_)
      LogDebug("SiStripHitEfficiency:HitEff")
          << "starting checking good event with < " << trackMultiplicityCut_ << " tracks" << endl;

    EventTrackCKF++;

#ifdef ExtendedCALIBTree
    //get dEdx info if available
    edm::Handle<ValueMap<DeDxData> > dEdxUncalibHandle;
    if (e.getByLabel("dedxMedianCTF", dEdxUncalibHandle)) {
      const ValueMap<DeDxData> dEdxTrackUncalib = *dEdxUncalibHandle.product();

      reco::TrackRef itTrack = reco::TrackRef(trackCollectionCKF, 0);
      dedx = dEdxTrackUncalib[itTrack].dEdx();
      dedxNOM = dEdxTrackUncalib[itTrack].numberOfMeasurements();
    } else {
      dedx = -999.0;
      dedxNOM = -999;
    }

    //get muon and ecal timing info if available
    edm::Handle<MuonCollection> muH;
    if (e.getByLabel("muonsWitht0Correction", muH)) {
      const MuonCollection& muonsT0 = *muH.product();
      if (!muonsT0.empty()) {
        MuonTime mt0 = muonsT0[0].time();
        timeDT = mt0.timeAtIpInOut;
        timeDTErr = mt0.timeAtIpInOutErr;
        timeDTDOF = mt0.nDof;

        bool hasCaloEnergyInfo = muonsT0[0].isEnergyValid();
        if (hasCaloEnergyInfo)
          timeECAL = muonsT0[0].calEnergy().ecal_time;
      }
    } else {
      timeDT = -999.0;
      timeDTErr = -999.0;
      timeDTDOF = -999;
      timeECAL = -999.0;
    }

#endif
    // actually should do a loop over all the tracks in the event here

    // Looping over traj-track associations to be able to get traj & track informations
    for (TrajTrackAssociationCollection::const_iterator it = trajTrackAssociationHandle->begin();
         it != trajTrackAssociationHandle->end();
         it++) {
      edm::Ref<std::vector<Trajectory> > itraj = it->key;
      reco::TrackRef itrack = it->val;

      // for each track, fill some variables such as number of hits and momentum
      nHits = itraj->foundHits();
#ifdef ExtendedCALIBTree
      nLostHits = itraj->lostHits();
      chi2 = (itraj->chiSquared() / itraj->ndof());
      p = itraj->lastMeasurement().updatedState().globalMomentum().mag();
#endif
      pT = sqrt((itraj->lastMeasurement().updatedState().globalMomentum().x() *
                 itraj->lastMeasurement().updatedState().globalMomentum().x()) +
                (itraj->lastMeasurement().updatedState().globalMomentum().y() *
                 itraj->lastMeasurement().updatedState().globalMomentum().y()));

      // track quality
      highPurity = itrack->quality(reco::TrackBase::TrackQuality::highPurity);

      std::vector<TrajectoryMeasurement> TMeas = itraj->measurements();
      totalNbHits += int(TMeas.size());
      vector<TrajectoryMeasurement>::iterator itm;
      double xloc = 0.;
      double yloc = 0.;
      double xErr = 0.;
      double yErr = 0.;
      double angleX = -999.;
      double angleY = -999.;
      double xglob, yglob, zglob;

      // Check whether the trajectory has some missing hits
      bool hasMissingHits = false;
      int previous_layer = 999;
      vector<unsigned int> missedLayers;
      for (const auto& itm : TMeas) {
        auto theHit = itm.recHit();
        unsigned int iidd = theHit->geographicalId().rawId();
        int layer = ::checkLayer(iidd, tTopo);
        int missedLayer = (layer + 1);
        int diffPreviousLayer = (layer - previous_layer);
        if (doMissingHitsRecovery_) {
          //Layers from TIB + TOB
          if (diffPreviousLayer == -2 && missedLayer > k_LayersStart && missedLayer < k_LayersAtTOBEnd) {
            missHitPerLayer[missedLayer] += 1;
            hasMissingHits = true;
          }
          //Layers from TID
          else if (diffPreviousLayer == -2 && (missedLayer > k_LayersAtTOBEnd + 1 && missedLayer <= k_LayersAtTIDEnd)) {
            missHitPerLayer[missedLayer] += 1;
            hasMissingHits = true;
          }
          //Layers from TEC
          else if (diffPreviousLayer == -2 && missedLayer > k_LayersAtTIDEnd && missedLayer <= k_LayersAtTECEnd) {
            missHitPerLayer[missedLayer] += 1;
            hasMissingHits = true;
          }

          //##### TID Layer 11 in transition TID -> TIB : layer is in TIB, previous layer  = 12
          if ((layer > k_LayersStart && layer <= k_LayersAtTIBEnd) && (previous_layer == 12)) {
            missHitPerLayer[11] += 1;
            hasMissingHits = true;
          }

          //##### TEC Layer 14 in transition TEC -> TOB : layer is in TOB, previous layer =  15
          if ((layer > k_LayersAtTIBEnd && layer <= k_LayersAtTOBEnd) && (previous_layer == 15)) {
            missHitPerLayer[14] += 1;
            hasMissingHits = true;
          }
        }
        if (theHit->getType() == TrackingRecHit::Type::missing)
          hasMissingHits = true;

        if (hasMissingHits)
          missedLayers.push_back(layer);
        previous_layer = layer;
      }

      // Loop on each measurement and take it into consideration
      //--------------------------------------------------------
      unsigned int prev_TKlayers = 0;
      for (itm = TMeas.begin(); itm != TMeas.end(); itm++) {
        auto theInHit = (*itm).recHit();

        LogDebug("SiStripHitEfficiency:HitEff") << "theInHit is valid = " << theInHit->isValid() << endl;

        unsigned int iidd = theInHit->geographicalId().rawId();

        unsigned int TKlayers = ::checkLayer(iidd, tTopo);
        LogDebug("SiStripHitEfficiency:HitEff") << "TKlayer from trajectory: " << TKlayers << "  from module = " << iidd
                                                << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/"
                                                << ((iidd & 0x3) == 1) << "/" << ((iidd & 0x3) == 2) << endl;

        // Test first and last points of the trajectory
        // the list of measurements starts from outer layers  !!! This could change -> should add a check
        bool isFirstMeas = (itm == (TMeas.end() - 1));
        bool isLastMeas = (itm == (TMeas.begin()));

        if (!useFirstMeas_ && isFirstMeas)
          continue;
        if (!useLastMeas_ && isLastMeas)
          continue;

        // In case of missing hit in the track, check whether to use the other hits or not.
        if (hasMissingHits && theInHit->getType() != TrackingRecHit::Type::missing &&
            !useAllHitsFromTracksWithMissingHits_)
          continue;

        // If Trajectory measurement from TOB 6 or TEC 9, skip it because it's always valid they are filled later
        if (TKlayers == 10 || TKlayers == 22) {
          LogDebug("SiStripHitEfficiency:HitEff") << "skipping original TM for TOB 6 or TEC 9" << endl;
          continue;
        }

        // Make vector of TrajectoryAtInvalidHits to hold the trajectories
        std::vector<TrajectoryAtInvalidHit> TMs;

        // Make AnalyticalPropagator to use in TAVH constructor
        AnalyticalPropagator propagator(magField_, anyDirection);

        // for double sided layers check both sensors--if no hit was found on either sensor surface,
        // the trajectory measurements only have one invalid hit entry on the matched surface
        // so get the TrajectoryAtInvalidHit for both surfaces and include them in the study
        if (::isDoubleSided(iidd, tTopo) && ((iidd & 0x3) == 0)) {
          // do hit eff check twice--once for each sensor
          //add a TM for each surface
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 1));
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 2));
        } else if (::isDoubleSided(iidd, tTopo) && (!::check2DPartner(iidd, TMeas))) {
          // if only one hit was found the trajectory measurement is on that sensor surface, and the other surface from
          // the matched layer should be added to the study as well
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 1));
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 2));
          LogDebug("SiStripHitEfficiency:HitEff") << " found a hit with a missing partner";
        } else {
          //only add one TM for the single surface and the other will be added in the next iteration
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator));
        }
        bool missingHitAdded = false;

        vector<TrajectoryMeasurement> tmpTmeas;
        unsigned int misLayer = TKlayers + 1;
        //Use bool doMissingHitsRecovery to add possible missing hits based on actual/previous hit
        if (doMissingHitsRecovery_) {
          if (int(TKlayers) - int(prev_TKlayers) == -2) {
            const DetLayer* detlayer = itm->layer();
            const LayerMeasurements layerMeasurements{*measurementTrackerHandle, *measurementTrackerEvent};
            const TrajectoryStateOnSurface tsos = itm->updatedState();
            std::vector<DetLayer::DetWithState> compatDets = detlayer->compatibleDets(tsos, *thePropagator, *estimator);

            if (misLayer > k_LayersAtTIDEnd && misLayer < k_LayersAtTECEnd) {  //TEC
              std::vector<ForwardDetLayer const*> negTECLayers =
                  measurementTrackerHandle->geometricSearchTracker()->negTecLayers();
              std::vector<ForwardDetLayer const*> posTECLayers =
                  measurementTrackerHandle->geometricSearchTracker()->posTecLayers();
              const DetLayer* tecLayerneg = negTECLayers[misLayer - k_LayersAtTIDEnd - 1];
              const DetLayer* tecLayerpos = posTECLayers[misLayer - k_LayersAtTIDEnd - 1];
              if (tTopo->tecSide(iidd) == 1) {
                tmpTmeas = layerMeasurements.measurements(*tecLayerneg, tsos, *thePropagator, *estimator);
              } else if (tTopo->tecSide(iidd) == 2) {
                tmpTmeas = layerMeasurements.measurements(*tecLayerpos, tsos, *thePropagator, *estimator);
              }
            }

            else if (misLayer == (k_LayersAtTIDEnd - 1) ||
                     misLayer == k_LayersAtTIDEnd) {  // This is for TID layers 12 and 13

              std::vector<ForwardDetLayer const*> negTIDLayers =
                  measurementTrackerHandle->geometricSearchTracker()->negTidLayers();
              std::vector<ForwardDetLayer const*> posTIDLayers =
                  measurementTrackerHandle->geometricSearchTracker()->posTidLayers();
              const DetLayer* tidLayerneg = negTIDLayers[misLayer - k_LayersAtTOBEnd - 1];
              const DetLayer* tidLayerpos = posTIDLayers[misLayer - k_LayersAtTOBEnd - 1];

              if (tTopo->tidSide(iidd) == 1) {
                tmpTmeas = layerMeasurements.measurements(*tidLayerneg, tsos, *thePropagator, *estimator);
              } else if (tTopo->tidSide(iidd) == 2) {
                tmpTmeas = layerMeasurements.measurements(*tidLayerpos, tsos, *thePropagator, *estimator);
              }
            }

            if (misLayer > k_LayersStart && misLayer < k_LayersAtTOBEnd) {  // Barrel

              std::vector<BarrelDetLayer const*> barrelTIBLayers =
                  measurementTrackerHandle->geometricSearchTracker()->tibLayers();
              std::vector<BarrelDetLayer const*> barrelTOBLayers =
                  measurementTrackerHandle->geometricSearchTracker()->tobLayers();

              if (misLayer > k_LayersStart && misLayer <= k_LayersAtTIBEnd) {
                const DetLayer* tibLayer = barrelTIBLayers[misLayer - k_LayersStart - 1];
                tmpTmeas = layerMeasurements.measurements(*tibLayer, tsos, *thePropagator, *estimator);
              } else if (misLayer > k_LayersAtTIBEnd && misLayer < k_LayersAtTOBEnd) {
                const DetLayer* tobLayer = barrelTOBLayers[misLayer - k_LayersAtTIBEnd - 1];
                tmpTmeas = layerMeasurements.measurements(*tobLayer, tsos, *thePropagator, *estimator);
              }
            }
          }
          if ((int(TKlayers) > k_LayersStart && int(TKlayers) <= k_LayersAtTIBEnd) && int(prev_TKlayers) == 12) {
            const DetLayer* detlayer = itm->layer();
            const LayerMeasurements layerMeasurements{*measurementTrackerHandle, *measurementTrackerEvent};
            const TrajectoryStateOnSurface tsos = itm->updatedState();
            std::vector<DetLayer::DetWithState> compatDets = detlayer->compatibleDets(tsos, *thePropagator, *estimator);
            std::vector<ForwardDetLayer const*> negTIDLayers =
                measurementTrackerHandle->geometricSearchTracker()->negTidLayers();
            std::vector<ForwardDetLayer const*> posTIDLayers =
                measurementTrackerHandle->geometricSearchTracker()->posTidLayers();

            const DetLayer* tidLayerneg = negTIDLayers[k_LayersStart];
            const DetLayer* tidLayerpos = posTIDLayers[k_LayersStart];
            if (tTopo->tidSide(iidd) == 1) {
              tmpTmeas = layerMeasurements.measurements(*tidLayerneg, tsos, *thePropagator, *estimator);
            } else if (tTopo->tidSide(iidd) == 2) {
              tmpTmeas = layerMeasurements.measurements(*tidLayerpos, tsos, *thePropagator, *estimator);
            }
          }

          if ((int(TKlayers) > k_LayersAtTIBEnd && int(TKlayers) <= k_LayersAtTOBEnd) && int(prev_TKlayers) == 15) {
            const DetLayer* detlayer = itm->layer();
            const LayerMeasurements layerMeasurements{*measurementTrackerHandle, *measurementTrackerEvent};
            const TrajectoryStateOnSurface tsos = itm->updatedState();
            std::vector<DetLayer::DetWithState> compatDets = detlayer->compatibleDets(tsos, *thePropagator, *estimator);

            std::vector<ForwardDetLayer const*> negTECLayers =
                measurementTrackerHandle->geometricSearchTracker()->negTecLayers();
            std::vector<ForwardDetLayer const*> posTECLayers =
                measurementTrackerHandle->geometricSearchTracker()->posTecLayers();

            const DetLayer* tecLayerneg = negTECLayers[k_LayersStart];
            const DetLayer* tecLayerpos = posTECLayers[k_LayersStart];
            if (tTopo->tecSide(iidd) == 1) {
              tmpTmeas = layerMeasurements.measurements(*tecLayerneg, tsos, *thePropagator, *estimator);
            } else if (tTopo->tecSide(iidd) == 2) {
              tmpTmeas = layerMeasurements.measurements(*tecLayerpos, tsos, *thePropagator, *estimator);
            }
          }

          if (!tmpTmeas.empty()) {
            TrajectoryMeasurement TM_tmp(tmpTmeas.back());
            unsigned int iidd_tmp = TM_tmp.recHit()->geographicalId().rawId();
            if (iidd_tmp != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << " hit actually being added to TM vector";
              if ((!useAllHitsFromTracksWithMissingHits_ || (!useFirstMeas_ && isFirstMeas)))
                TMs.clear();
              if (::isDoubleSided(iidd_tmp, tTopo)) {
                TMs.push_back(TrajectoryAtInvalidHit(TM_tmp, tTopo, tkgeom, propagator, 1));
                TMs.push_back(TrajectoryAtInvalidHit(TM_tmp, tTopo, tkgeom, propagator, 2));
              } else
                TMs.push_back(TrajectoryAtInvalidHit(TM_tmp, tTopo, tkgeom, propagator));
              missingHitAdded = true;
              hitRecoveryCounters[misLayer] += 1;
            }
          }
        }

        prev_TKlayers = TKlayers;
        if (!useFirstMeas_ && isFirstMeas && !missingHitAdded)
          continue;
        if (!useLastMeas_ && isLastMeas)
          continue;
        bool hitsWithBias = false;
        for (auto ilayer : missedLayers) {
          if (ilayer < TKlayers)
            hitsWithBias = true;
        }
        if (hasMissingHits && theInHit->getType() != TrackingRecHit::Type::missing && !missingHitAdded &&
            hitsWithBias && !useAllHitsFromTracksWithMissingHits_) {
          continue;
        }
        //////////////////////////////////////////////
        //Now check for tracks at TOB6 and TEC9

        // to make sure we only propagate on the last TOB5 hit check the next entry isn't also in TOB5
        // to avoid bias, make sure the TOB5 hit is valid (an invalid hit on TOB5 could only exist with a valid hit on TOB6)

        bool isValid = theInHit->isValid();
        bool isLast = (itm == (TMeas.end() - 1));
        bool isLastTOB5 = true;
        if (!isLast) {
          if (::checkLayer((++itm)->recHit()->geographicalId().rawId(), tTopo) == 9)
            isLastTOB5 = false;
          else
            isLastTOB5 = true;
          --itm;
        }

        if (TKlayers == 9 && isValid && isLastTOB5) {
          //	  if ( TKlayers==9 && itm==TMeas.rbegin()) {
          //	  if ( TKlayers==9 && (itm==TMeas.back()) ) {	  // to check for only the last entry in the trajectory for propagation
          std::vector<BarrelDetLayer const*> barrelTOBLayers =
              measurementTrackerHandle->geometricSearchTracker()->tobLayers();
          const DetLayer* tob6 = barrelTOBLayers[barrelTOBLayers.size() - 1];
          const LayerMeasurements layerMeasurements{*measurementTrackerHandle, *measurementTrackerEvent};
          const TrajectoryStateOnSurface tsosTOB5 = itm->updatedState();
          auto tmp = layerMeasurements.measurements(*tob6, tsosTOB5, *thePropagator, *estimator);

          if (!tmp.empty()) {
            LogDebug("SiStripHitEfficiency:HitEff") << "size of TM from propagation = " << tmp.size() << endl;

            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            TrajectoryMeasurement tob6TM(tmp.back());
            const auto& tob6Hit = tob6TM.recHit();

            if (tob6Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << "tob6 hit actually being added to TM vector" << endl;
              TMs.push_back(TrajectoryAtInvalidHit(tob6TM, tTopo, tkgeom, propagator));
            }
          }
        }

        bool isLastTEC8 = true;
        if (!isLast) {
          if (::checkLayer((++itm)->recHit()->geographicalId().rawId(), tTopo) == 21)
            isLastTEC8 = false;
          else
            isLastTEC8 = true;
          --itm;
        }

        if (TKlayers == 21 && isValid && isLastTEC8) {
          std::vector<const ForwardDetLayer*> posTecLayers =
              measurementTrackerHandle->geometricSearchTracker()->posTecLayers();
          const DetLayer* tec9pos = posTecLayers[posTecLayers.size() - 1];
          std::vector<const ForwardDetLayer*> negTecLayers =
              measurementTrackerHandle->geometricSearchTracker()->negTecLayers();
          const DetLayer* tec9neg = negTecLayers[negTecLayers.size() - 1];
          const LayerMeasurements layerMeasurements{*measurementTrackerHandle, *measurementTrackerEvent};
          const TrajectoryStateOnSurface tsosTEC9 = itm->updatedState();

          // check if track on positive or negative z
          if (!(iidd == StripSubdetector::TEC))
            LogDebug("SiStripHitEfficiency:HitEff") << "there is a problem with TEC 9 extrapolation" << endl;

          //LOGPRINT << " tec9 id = " << iidd << " and side = " << tTopo->tecSide(iidd) << endl;
          vector<TrajectoryMeasurement> tmp;
          if (tTopo->tecSide(iidd) == 1) {
            tmp = layerMeasurements.measurements(*tec9neg, tsosTEC9, *thePropagator, *estimator);
            //LOGPRINT << "on negative side" << endl;
          }
          if (tTopo->tecSide(iidd) == 2) {
            tmp = layerMeasurements.measurements(*tec9pos, tsosTEC9, *thePropagator, *estimator);
            //LOGPRINT << "on positive side" << endl;
          }

          if (!tmp.empty()) {
            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            TrajectoryMeasurement tec9TM(tmp.back());
            const auto& tec9Hit = tec9TM.recHit();

            unsigned int tec9id = tec9Hit->geographicalId().rawId();
            LogDebug("SiStripHitEfficiency:HitEff")
                << "tec9id = " << tec9id << " is Double sided = " << ::isDoubleSided(tec9id, tTopo)
                << "  and 0x3 = " << (tec9id & 0x3) << endl;

            if (tec9Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << "tec9 hit actually being added to TM vector" << endl;
              // in tec the hit can be single or doubled sided. whenever the invalid hit at the end of vector of TMs is
              // double sided it is always on the matched surface, so we need to split it into the true sensor surfaces
              if (::isDoubleSided(tec9id, tTopo)) {
                TMs.push_back(TrajectoryAtInvalidHit(tec9TM, tTopo, tkgeom, propagator, 1));
                TMs.push_back(TrajectoryAtInvalidHit(tec9TM, tTopo, tkgeom, propagator, 2));
              } else
                TMs.push_back(TrajectoryAtInvalidHit(tec9TM, tTopo, tkgeom, propagator));
            }
          }  //else LOGPRINT << "tec9 tmp empty" << endl;
        }
        hitTotalCounters[TKlayers] += 1;

        ////////////////////////////////////////////////////////

        // Modules Constraints

        for (std::vector<TrajectoryAtInvalidHit>::const_iterator TM = TMs.begin(); TM != TMs.end(); ++TM) {
          // --> Get trajectory from combinatedState
          iidd = TM->monodet_id();
          LogDebug("SiStripHitEfficiency:HitEff") << "setting iidd = " << iidd << " before checking efficiency and ";

          xloc = TM->localX();
          yloc = TM->localY();

          angleX = atan(TM->localDxDz());
          angleY = atan(TM->localDyDz());

          TrajLocErrX = 0.0;
          TrajLocErrY = 0.0;

          xglob = TM->globalX();
          yglob = TM->globalY();
          zglob = TM->globalZ();
          xErr = TM->localErrorX();
          yErr = TM->localErrorY();

          TrajGlbX = 0.0;
          TrajGlbY = 0.0;
          TrajGlbZ = 0.0;
          withinAcceptance = TM->withinAcceptance();

          trajHitValid = TM->validHit();
          int TrajStrip = -1;

          // reget layer from iidd here, to account for TOB 6 and TEC 9 TKlayers being off
          TKlayers = ::checkLayer(iidd, tTopo);

          if ((layers == TKlayers) || (layers == 0)) {  // Look at the layer not used to reconstruct the track
            whatlayer = TKlayers;
            LogDebug("SiStripHitEfficiency:HitEff") << "Looking at layer under study" << endl;
            ModIsBad = 2;
            Id = 0;
            SiStripQualBad = 0;
            run = 0;
            event = 0;
            TrajLocX = 0.0;
            TrajLocY = 0.0;
            TrajLocAngleX = -999.0;
            TrajLocAngleY = -999.0;
            ResX = 0.0;
            ResXSig = 0.0;
            ClusterLocX = 0.0;
            ClusterLocY = 0.0;
            ClusterLocErrX = 0.0;
            ClusterLocErrY = 0.0;
            ClusterStoN = 0.0;
            bunchx = 0;
            commonMode = -100;

            // RPhi RecHit Efficiency

            if (!input.empty()) {
              LogDebug("SiStripHitEfficiency:HitEff") << "Checking clusters with size = " << input.size() << endl;
              int nClusters = 0;
              std::vector<std::vector<float> >
                  VCluster_info;  //fill with X residual, X residual pull, local X, sig(X), local Y, sig(Y), StoN
              for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter = input.begin(); DSViter != input.end();
                   DSViter++) {
                // DSViter is a vector of SiStripClusters located on a single module
                //if (DEBUG)      LOGPRINT << "the ID from the DSViter = " << DSViter->id() << endl;
                unsigned int ClusterId = DSViter->id();
                if (ClusterId == iidd) {
                  LogDebug("SiStripHitEfficiency:HitEff")
                      << "found  (ClusterId == iidd) with ClusterId = " << ClusterId << " and iidd = " << iidd << endl;
                  DetId ClusterDetId(ClusterId);
                  const StripGeomDetUnit* stripdet = (const StripGeomDetUnit*)tkgeom->idToDetUnit(ClusterDetId);
                  const StripTopology& Topo = stripdet->specificTopology();

                  float hbedge = 0.0;
                  float htedge = 0.0;
                  float hapoth = 0.0;
                  float uylfac = 0.0;
                  float uxlden = 0.0;
                  if (TKlayers >= 11) {
                    const BoundPlane& plane = stripdet->surface();
                    const TrapezoidalPlaneBounds* trapezoidalBounds(
                        dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
                    std::array<const float, 4> const& parameterTrap =
                        (*trapezoidalBounds).parameters();  // el bueno aqui
                    hbedge = parameterTrap[0];
                    htedge = parameterTrap[1];
                    hapoth = parameterTrap[3];
                    uylfac = (htedge - hbedge) / (htedge + hbedge) / hapoth;
                    uxlden = 1 + yloc * uylfac;
                  }

                  // Need to know position of trajectory in strip number for selecting the right APV later
                  if (TrajStrip == -1) {
                    int nstrips = Topo.nstrips();
                    float pitch = stripdet->surface().bounds().width() / nstrips;
                    TrajStrip = xloc / pitch + nstrips / 2.0;
                    // Need additionnal corrections for endcap
                    if (TKlayers >= 11) {
                      float TrajLocXMid = xloc / (1 + (htedge - hbedge) * yloc / (htedge + hbedge) /
                                                          hapoth);  // radialy extrapolated x loc position at middle
                      TrajStrip = TrajLocXMid / pitch + nstrips / 2.0;
                    }
                    //LOGPRINT<<" Layer "<<TKlayers<<" TrajStrip: "<<nstrips<<" "<<pitch<<" "<<TrajStrip<<endl;
                  }

                  for (edmNew::DetSet<SiStripCluster>::const_iterator iter = DSViter->begin(); iter != DSViter->end();
                       ++iter) {
                    //iter is a single SiStripCluster
                    StripClusterParameterEstimator::LocalValues parameters = stripcpe.localParameters(*iter, *stripdet);
                    float res = (parameters.first.x() - xloc);
                    float sigma = ::checkConsistency(parameters, xloc, xErr);
                    // The consistency is probably more accurately measured with the Chi2MeasurementEstimator. To use it
                    // you need a TransientTrackingRecHit instead of the cluster
                    //theEstimator=       new Chi2MeasurementEstimator(30);
                    //const Chi2MeasurementEstimator *theEstimator(100);
                    //theEstimator->estimate(TM->tsos(), TransientTrackingRecHit);

                    if (TKlayers >= 11) {
                      res = parameters.first.x() - xloc / uxlden;  // radialy extrapolated x loc position at middle
                      sigma = abs(res) /
                              sqrt(parameters.second.xx() + xErr * xErr / uxlden / uxlden +
                                   yErr * yErr * xloc * xloc * uylfac * uylfac / uxlden / uxlden / uxlden / uxlden);
                    }

                    siStripClusterInfo_.setCluster(*iter, ClusterId);
                    // signal to noise from SiStripClusterInfo not working in 225. I'll fix this after the interface
                    // redesign in 300 -ku
                    //float cluster_info[7] = {res, sigma, parameters.first.x(), sqrt(parameters.second.xx()), parameters.first.y(), sqrt(parameters.second.yy()), signal_to_noise};
                    std::vector<float> cluster_info;
                    cluster_info.push_back(res);
                    cluster_info.push_back(sigma);
                    cluster_info.push_back(parameters.first.x());
                    cluster_info.push_back(sqrt(parameters.second.xx()));
                    cluster_info.push_back(parameters.first.y());
                    cluster_info.push_back(sqrt(parameters.second.yy()));
                    cluster_info.push_back(siStripClusterInfo_.signalOverNoise());
                    VCluster_info.push_back(cluster_info);
                    nClusters++;
                    LogDebug("SiStripHitEfficiency:HitEff") << "Have ID match. residual = " << VCluster_info.back()[0]
                                                            << "  res sigma = " << VCluster_info.back()[1] << endl;
                    LogDebug("SiStripHitEfficiency:HitEff")
                        << "trajectory measurement compatability estimate = " << (*itm).estimate() << endl;
                    LogDebug("SiStripHitEfficiency:HitEff")
                        << "hit position = " << parameters.first.x() << "  hit error = " << sqrt(parameters.second.xx())
                        << "  trajectory position = " << xloc << "  traj error = " << xErr << endl;
                  }
                }
              }
              float FinalResSig = 1000.0;
              float FinalCluster[7] = {1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0};
              if (nClusters > 0) {
                LogDebug("SiStripHitEfficiency:HitEff") << "found clusters > 0" << endl;
                if (nClusters > 1) {
                  //get the smallest one
                  vector<vector<float> >::iterator ires;
                  for (ires = VCluster_info.begin(); ires != VCluster_info.end(); ires++) {
                    if (abs((*ires)[1]) < abs(FinalResSig)) {
                      FinalResSig = (*ires)[1];
                      for (unsigned int i = 0; i < ires->size(); i++) {
                        LogDebug("SiStripHitEfficiency:HitEff")
                            << "filling final cluster. i = " << i << " before fill FinalCluster[i]=" << FinalCluster[i]
                            << " and (*ires)[i] =" << (*ires)[i] << endl;
                        FinalCluster[i] = (*ires)[i];
                        LogDebug("SiStripHitEfficiency:HitEff")
                            << "filling final cluster. i = " << i << " after fill FinalCluster[i]=" << FinalCluster[i]
                            << " and (*ires)[i] =" << (*ires)[i] << endl;
                      }
                    }
                    LogDebug("SiStripHitEfficiency:HitEff")
                        << "iresidual = " << (*ires)[0] << "  isigma = " << (*ires)[1]
                        << "  and FinalRes = " << FinalCluster[0] << endl;
                  }
                } else {
                  FinalResSig = VCluster_info.at(0)[1];
                  for (unsigned int i = 0; i < VCluster_info.at(0).size(); i++) {
                    FinalCluster[i] = VCluster_info.at(0)[i];
                  }
                }
                VCluster_info.clear();
              }

              LogDebug("SiStripHitEfficiency:HitEff")
                  << "Final residual in X = " << FinalCluster[0] << "+-" << (FinalCluster[0] / FinalResSig) << endl;
              LogDebug("SiStripHitEfficiency:HitEff") << "Checking location of trajectory: abs(yloc) = " << abs(yloc)
                                                      << "  abs(xloc) = " << abs(xloc) << endl;
              LogDebug("SiStripHitEfficiency:HitEff")
                  << "Checking location of cluster hit: yloc = " << FinalCluster[4] << "+-" << FinalCluster[5]
                  << "  xloc = " << FinalCluster[2] << "+-" << FinalCluster[3] << endl;
              LogDebug("SiStripHitEfficiency:HitEff") << "Final cluster signal to noise = " << FinalCluster[6] << endl;

              float exclusionWidth = 0.4;
              float TOBexclusion = 0.0;
              float TECexRing5 = -0.89;
              float TECexRing6 = -0.56;
              float TECexRing7 = 0.60;
              //Added by Chris Edelmaier to do TEC bonding exclusion
              int subdetector = ((iidd >> 25) & 0x7);
              int ringnumber = ((iidd >> 5) & 0x7);

              //New TOB and TEC bonding region exclusion zone
              if ((TKlayers >= 5 && TKlayers < 11) ||
                  ((subdetector == 6) && ((ringnumber >= 5) && (ringnumber <= 7)))) {
                //There are only 2 cases that we need to exclude for
                float highzone = 0.0;
                float lowzone = 0.0;
                float higherr = yloc + 5.0 * yErr;
                float lowerr = yloc - 5.0 * yErr;
                if (TKlayers >= 5 && TKlayers < 11) {
                  //TOB zone
                  highzone = TOBexclusion + exclusionWidth;
                  lowzone = TOBexclusion - exclusionWidth;
                } else if (ringnumber == 5) {
                  //TEC ring 5
                  highzone = TECexRing5 + exclusionWidth;
                  lowzone = TECexRing5 - exclusionWidth;
                } else if (ringnumber == 6) {
                  //TEC ring 6
                  highzone = TECexRing6 + exclusionWidth;
                  lowzone = TECexRing6 - exclusionWidth;
                } else if (ringnumber == 7) {
                  //TEC ring 7
                  highzone = TECexRing7 + exclusionWidth;
                  lowzone = TECexRing7 - exclusionWidth;
                }
                //Now that we have our exclusion region, we just have to properly identify it
                if ((highzone <= higherr) && (highzone >= lowerr))
                  withinAcceptance = false;
                if ((lowzone >= lowerr) && (lowzone <= higherr))
                  withinAcceptance = false;
                if ((higherr <= highzone) && (higherr >= lowzone))
                  withinAcceptance = false;
                if ((lowerr >= lowzone) && (lowerr <= highzone))
                  withinAcceptance = false;
              }

              // fill ntuple varibles
              //get global position from module id number iidd
              TrajGlbX = xglob;
              TrajGlbY = yglob;
              TrajGlbZ = zglob;

              TrajLocErrX = xErr;
              TrajLocErrY = yErr;

              Id = iidd;
              run = run_nr;
              event = ev_nr;
              bunchx = bunch_nr;
              //if ( SiStripQuality_->IsModuleBad(iidd) ) {
              if (SiStripQuality_->getBadApvs(iidd) != 0) {
                SiStripQualBad = 1;
                LogDebug("SiStripHitEfficiency:HitEff") << "strip is bad from SiStripQuality" << endl;
              } else {
                SiStripQualBad = 0;
                LogDebug("SiStripHitEfficiency:HitEff") << "strip is good from SiStripQuality" << endl;
              }

              //check for FED-detected errors and include those in SiStripQualBad
              for (unsigned int ii = 0; ii < fedErrorIds.size(); ii++) {
                if (iidd == fedErrorIds[ii].rawId())
                  SiStripQualBad = 1;
              }

              TrajLocX = xloc;
              TrajLocY = yloc;
              TrajLocAngleX = angleX;
              TrajLocAngleY = angleY;
              ResX = FinalCluster[0];
              ResXSig = FinalResSig;
              if (FinalResSig != FinalCluster[1])
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "Problem with best cluster selection because FinalResSig = " << FinalResSig
                    << " and FinalCluster[1] = " << FinalCluster[1] << endl;
              ClusterLocX = FinalCluster[2];
              ClusterLocY = FinalCluster[4];
              ClusterLocErrX = FinalCluster[3];
              ClusterLocErrY = FinalCluster[5];
              ClusterStoN = FinalCluster[6];

              // CM of APV crossed by traj
              if (addCommonMode_)
                if (commonModeDigis.isValid() && TrajStrip >= 0 && TrajStrip <= 768) {
                  edm::DetSetVector<SiStripRawDigi>::const_iterator digiframe = commonModeDigis->find(iidd);
                  if (digiframe != commonModeDigis->end())
                    if ((unsigned)TrajStrip / 128 < digiframe->data.size())
                      commonMode = digiframe->data.at(TrajStrip / 128).adc();
                }

              LogDebug("SiStripHitEfficiency:HitEff") << "before check good" << endl;

              if (FinalResSig < 999.0) {  //could make requirement on track/hit consistency, but for
                //now take anything with a hit on the module
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "hit being counted as good " << FinalCluster[0] << " FinalRecHit " << iidd << "   TKlayers  "
                    << TKlayers << " xloc " << xloc << " yloc  " << yloc << " module " << iidd
                    << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/" << ((iidd & 0x3) == 1) << "/"
                    << ((iidd & 0x3) == 2) << endl;
                ModIsBad = 0;
                traj->Fill();
              } else {
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "hit being counted as bad   ######### Invalid RPhi FinalResX " << FinalCluster[0]
                    << " FinalRecHit " << iidd << "   TKlayers  " << TKlayers << " xloc " << xloc << " yloc  " << yloc
                    << " module " << iidd << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/"
                    << ((iidd & 0x3) == 1) << "/" << ((iidd & 0x3) == 2) << endl;
                ModIsBad = 1;
                traj->Fill();

                LogDebug("SiStripHitEfficiency:HitEff") << " RPhi Error " << sqrt(xErr * xErr + yErr * yErr)
                                                        << " ErrorX " << xErr << " yErr " << yErr << endl;
              }
              LogDebug("SiStripHitEfficiency:HitEff") << "after good location check" << endl;
            }
            LogDebug("SiStripHitEfficiency:HitEff") << "after list of clusters" << endl;
          }
          LogDebug("SiStripHitEfficiency:HitEff") << "After layers=TKLayers if" << endl;
        }
        LogDebug("SiStripHitEfficiency:HitEff") << "After looping over TrajAtValidHit list" << endl;
      }
      LogDebug("SiStripHitEfficiency:HitEff") << "end TMeasurement loop" << endl;
    }
    LogDebug("SiStripHitEfficiency:HitEff") << "end of trajectories loop" << endl;
  }
}

void HitEff::endJob() {
  traj->GetDirectory()->cd();
  traj->Write();

  LogDebug("SiStripHitEfficiency:HitEff") << " Events Analysed             " << events << endl;
  LogDebug("SiStripHitEfficiency:HitEff") << " Number Of Tracked events    " << EventTrackCKF << endl;

  if (doMissingHitsRecovery_) {
    float totTIB = 0.0;
    float totTOB = 0.0;
    float totTID = 0.0;
    float totTEC = 0.0;

    float totTIBrepro = 0.0;
    float totTOBrepro = 0.0;
    float totTIDrepro = 0.0;
    float totTECrepro = 0.0;

    edm::LogInfo("SiStripHitEfficiency:HitEff") << "Within TIB :";
    for (int i = 0; i <= k_LayersAtTIBEnd; i++) {
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Layer " << i << " has : " << missHitPerLayer[i] << "/" << totalNbHits << " = "
          << (missHitPerLayer[i] * 1.0 / totalNbHits) * 100 << " % of missing hit";
      totTIB += missHitPerLayer[i];
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Removing recovered hits : layer " << i << " has : " << missHitPerLayer[i] - hitRecoveryCounters[i] << "/"
          << totalNbHits << " = " << ((missHitPerLayer[i] - hitRecoveryCounters[i]) * 1.0 / totalNbHits) * 100
          << " % of missing hit";
      totTIBrepro += (missHitPerLayer[i] - hitRecoveryCounters[i]);
    }
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "TOTAL % of missing hits within TIB :" << (totTIB * 1.0 / totalNbHits) * 100 << "%";
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "AFTER repropagation :" << (totTIBrepro * 1.0 / totalNbHits) * 100 << "%";

    edm::LogInfo("SiStripHitEfficiency:HitEff") << "Within TOB :";
    for (int i = k_LayersAtTIBEnd + 1; i <= k_LayersAtTOBEnd; i++) {
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Layer " << i << " has : " << missHitPerLayer[i] << "/" << totalNbHits << " = "
          << (missHitPerLayer[i] * 1.0 / totalNbHits) * 100 << " % of missing hit";
      totTOB += missHitPerLayer[i];
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Removing recovered hits : layer " << i << " has : " << missHitPerLayer[i] - hitRecoveryCounters[i] << "/"
          << totalNbHits << " = " << ((missHitPerLayer[i] - hitRecoveryCounters[i]) * 1.0 / totalNbHits) * 100
          << " % of missing hit";
      totTOBrepro += (missHitPerLayer[i] - hitRecoveryCounters[i]);
    }
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "TOTAL % of missing hits within TOB :" << (totTOB * 1.0 / totalNbHits) * 100 << "%";
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "AFTER repropagation :" << (totTOBrepro * 1.0 / totalNbHits) * 100 << "%";

    edm::LogInfo("SiStripHitEfficiency:HitEff") << "Within TID :";
    for (int i = k_LayersAtTOBEnd + 1; i <= k_LayersAtTIDEnd; i++) {
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Layer " << i << " has : " << missHitPerLayer[i] << "/" << totalNbHits << " = "
          << (missHitPerLayer[i] * 1.0 / totalNbHits) * 100 << " % of missing hit";
      totTID += missHitPerLayer[i];
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Removing recovered hits : layer " << i << " has : " << missHitPerLayer[i] - hitRecoveryCounters[i] << "/"
          << totalNbHits << " = " << ((missHitPerLayer[i] - hitRecoveryCounters[i]) * 1.0 / totalNbHits) * 100
          << " % of missing hit";
      totTIDrepro += (missHitPerLayer[i] - hitRecoveryCounters[i]);
    }
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "TOTAL % of missing hits within TID :" << (totTID * 1.0 / totalNbHits) * 100 << "%";
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "AFTER repropagation :" << (totTIDrepro * 1.0 / totalNbHits) * 100 << "%";

    edm::LogInfo("SiStripHitEfficiency:HitEff") << "Within TEC :";
    for (int i = k_LayersAtTIDEnd + 1; i < k_END_OF_LAYERS; i++) {
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Layer " << i << " has : " << missHitPerLayer[i] << "/" << totalNbHits << " = "
          << (missHitPerLayer[i] * 1.0 / totalNbHits) * 100 << " % of missing hit";
      totTEC += missHitPerLayer[i];
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << "Removing recovered hits : layer " << i << " has : " << missHitPerLayer[i] - hitRecoveryCounters[i] << "/"
          << totalNbHits << " = " << ((missHitPerLayer[i] - hitRecoveryCounters[i]) * 1.0 / totalNbHits) * 100
          << " % of missing hit";
      totTECrepro += (missHitPerLayer[i] - hitRecoveryCounters[i]);
    }
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "TOTAL % of missing hits within TEC :" << (totTEC * 1.0 / totalNbHits) * 100 << "%";
    edm::LogInfo("SiStripHitEfficiency:HitEff")
        << "AFTER repropagation :" << (totTECrepro * 1.0 / totalNbHits) * 100 << "%";

    edm::LogInfo("SiStripHitEfficiency:HitEff") << " Hit recovery summary:";

    for (int ilayer = 0; ilayer < k_END_OF_LAYERS; ilayer++) {
      edm::LogInfo("SiStripHitEfficiency:HitEff")
          << " layer " << ilayer << ": " << hitRecoveryCounters[ilayer] << " / " << hitTotalCounters[ilayer];
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HitEff);

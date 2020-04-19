////////////////////////////////////////////////////////////////////////////////
// Package:          CalibTracker/SiStripHitEfficiency
// Class:            HitEff (CalibTree production
// Original Author:  Keith Ulmer--University of Colorado
//                   keith.ulmer@colorado.edu
// Class:            SiStripHitEffFromCalibTree
// Original Author:  Christopher Edelmaier
//
///////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


class SiStripHitEfficiencyWorker : public DQMEDAnalyzer {
public:
  explicit SiStripHitEfficiencyWorker(const edm::ParameterSet& conf);
  ~SiStripHitEfficiencyWorker() override;

private:
  void beginJob(); // TODO remove
  void endJob(); // TODO remove
  void bookHistograms(DQMStore::IBooker& booker, const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > commonModeToken_;

  SiStripClusterInfo siStripClusterInfo_;

  bool addLumi_;
  bool addCommonMode_;
  bool cutOnTracks_;
  unsigned int trackMultiplicityCut_;
  bool useFirstMeas_;
  bool useLastMeas_;
  bool useAllHitsFromTracksWithMissingHits_;

  const edm::EDGetTokenT<reco::TrackCollection> combinatorialTracks_token_;
  const edm::EDGetTokenT<std::vector<Trajectory> > trajectories_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAsso_token_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  const edm::EDGetTokenT<DetIdCollection> digis_token_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> trackerEvent_token_;

  edm::ParameterSet conf_;

  int events, EventTrackCKF;

  unsigned int layers;
  bool DEBUG;

// Tree declarations
  unsigned int whatlayer;
// Trajectory positions for modules included in the study
  float TrajGlbX, TrajGlbY, TrajGlbZ;
  float TrajLocX, TrajLocY;
  float ClusterLocX;
  float ResXSig;
  unsigned int ModIsBad;
  unsigned int Id;
  unsigned int SiStripQualBad;
  bool withinAcceptance;
  bool highPurity;
  unsigned int run, event, bunchx;
  float instLumi, PU;
  float commonMode;
  /* Used in SiStripHitEffFromCalibTree:
   * run              -> "run"              -> run
   * event            -> "event"            -> evt
   * ModIsBad         -> "ModIsBad"         -> isBad
   * SiStripQualBad   -> "SiStripQualBad""  -> quality
   * Id               -> "Id"               -> id
   * withinAcceptance -> "withinAcceptance" -> accept
   * whatlayer        -> "layer"            -> layer_wheel
   * highPurity       -> "highPurity"       -> highPurity
   * TrajGlbX         -> "TrajGlbX"         -> x
   * TrajGlbY         -> "TrajGlbY"         -> y
   * TrajGlbZ         -> "TrajGlbZ"         -> z
   * ResXSig          -> "ResXSig"          -> resxsig
   * TrajLocX         -> "TrajLocX"         -> TrajLocX
   * TrajLocY         -> "TrajLocY"         -> TrajLocY
   * ClusterLocX      -> "ClusterLocX"      -> ClusterLocX
   * bunchx           -> "bunchx"           -> bx
   * instLumi         -> "instLumi"         -> instLumi         ## if addLumi_
   * PU               -> "PU"               -> PU               ## if addLumi_
   * commonMode       -> "commonMode"       -> CM               ## if addCommonMode_ / _useCM
  */
};

//
// constructors and destructor
//

SiStripHitEfficiencyWorker::SiStripHitEfficiencyWorker(const edm::ParameterSet& conf)
    : scalerToken_(consumes<LumiScalersCollection>(conf.getParameter<edm::InputTag>("lumiScalers"))),
      commonModeToken_(mayConsume<edm::DetSetVector<SiStripRawDigi> >(conf.getParameter<edm::InputTag>("commonMode"))),
      siStripClusterInfo_(consumesCollector()),
      combinatorialTracks_token_(
          consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("combinatorialTracks"))),
      trajectories_token_(consumes<std::vector<Trajectory> >(conf.getParameter<edm::InputTag>("trajectories"))),
      trajTrackAsso_token_(consumes<TrajTrackAssociationCollection>(conf.getParameter<edm::InputTag>("trajectories"))),
      clusters_token_(
          consumes<edmNew::DetSetVector<SiStripCluster> >(conf.getParameter<edm::InputTag>("siStripClusters"))),
      digis_token_(consumes<DetIdCollection>(conf.getParameter<edm::InputTag>("siStripDigis"))),
      trackerEvent_token_(consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent"))),
      conf_(conf) {
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
}

// Virtual destructor needed.
SiStripHitEfficiencyWorker::~SiStripHitEfficiencyWorker() {}

void SiStripHitEfficiencyWorker::beginJob() {
  // TODO convert to counters, or simply remove?
  events = 0;
  EventTrackCKF = 0;
}

void SiStripHitEfficiencyWorker::bookHistograms(DQMStore::IBooker& booker, const edm::Run& run, const edm::EventSetup& setup) {

}

namespace {

double checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters, double xx, double xerr) {
  double error = sqrt(parameters.second.xx() + xerr * xerr);
  double separation = abs(parameters.first.x() - xx);
  double consistency = separation / error;
  return consistency;
}

bool isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) {
  StripSubdetector strip = StripSubdetector(iidd);
  unsigned int subid = strip.subdetId();
  unsigned int layer = 0;
  if (subid == StripSubdetector::TIB) {
    layer = tTopo->tibLayer(iidd);
    if (layer == 1 || layer == 2)
      return true;
    else
      return false;
  } else if (subid == StripSubdetector::TOB) {
    layer = tTopo->tobLayer(iidd) + 4;
    if (layer == 5 || layer == 6)
      return true;
    else
      return false;
  } else if (subid == StripSubdetector::TID) {
    layer = tTopo->tidRing(iidd) + 10;
    if (layer == 11 || layer == 12)
      return true;
    else
      return false;
  } else if (subid == StripSubdetector::TEC) {
    layer = tTopo->tecRing(iidd) + 13;
    if (layer == 14 || layer == 15 || layer == 18)
      return true;
    else
      return false;
  } else
    return false;
}

bool check2DPartner(unsigned int iidd, const std::vector<TrajectoryMeasurement>& traj) {
  unsigned int partner_iidd = 0;
  bool found2DPartner = false;
  // first get the id of the other detector
  if ((iidd & 0x3) == 1)
    partner_iidd = iidd + 1;
  if ((iidd & 0x3) == 2)
    partner_iidd = iidd - 1;
  // next look in the trajectory measurements for a measurement from that detector
  // loop through trajectory measurements to find the partner_iidd
  for ( const auto& tm : traj ) {
    if (tm.recHit()->geographicalId().rawId() == partner_iidd) {
      found2DPartner = true;
    }
  }
  return found2DPartner;
}

unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo) {
  StripSubdetector strip = StripSubdetector(iidd);
  unsigned int subid = strip.subdetId();
  if (subid == StripSubdetector::TIB) {
    return tTopo->tibLayer(iidd);
  }
  if (subid == StripSubdetector::TOB) {
    return tTopo->tobLayer(iidd) + 4;
  }
  if (subid == StripSubdetector::TID) {
    return tTopo->tidWheel(iidd) + 10;
  }
  if (subid == StripSubdetector::TEC) {
    return tTopo->tecWheel(iidd) + 13;
  }
  return 0;
}

bool isInBondingExclusionZone(unsigned int iidd, unsigned int TKlayers, double yloc, double yErr, const TrackerTopology* tTopo) {
  constexpr float exclusionWidth = 0.4;
  constexpr float TOBexclusion = 0.0;
  constexpr float TECexRing5 = -0.89;
  constexpr float TECexRing6 = -0.56;
  constexpr float TECexRing7 = 0.60;

  //Added by Chris Edelmaier to do TEC bonding exclusion
  const int subdetector = ((iidd >> 25) & 0x7);
  const int ringnumber = ((iidd >> 5) & 0x7);

  bool inZone = false;
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
      inZone = true;
    if ((lowzone >= lowerr) && (lowzone <= higherr))
      inZone = true;
    if ((higherr <= highzone) && (higherr >= lowzone))
      inZone = true;
    if ((lowerr >= lowzone) && (lowerr <= highzone))
      inZone = true;
  }
  return inZone;
}

} // anonymous namespace

void SiStripHitEfficiencyWorker::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  siStripClusterInfo_.initEvent(es);

  //  bool DEBUG = false;

  LogDebug("SiStripHitEfficiency:HitEff") << "beginning analyze from HitEff" << std::endl;

  // Step A: Get Inputs

  int run_nr = e.id().run();
  int ev_nr = e.id().event();
  int bunch_nr = e.bunchCrossing();

  // Luminosity informations
  edm::Handle<LumiScalersCollection> lumiScalers;
  instLumi = 0;
  PU = 0;
  if (addLumi_) {
    e.getByToken(scalerToken_, lumiScalers);
    if (lumiScalers->begin() != lumiScalers->end()) {
      instLumi = lumiScalers->begin()->instantLumi();
      PU = lumiScalers->begin()->pileup();
    }
  }

  edm::Handle<edm::DetSetVector<SiStripRawDigi> > commonModeDigis;
  if (addCommonMode_)
    e.getByToken(commonModeToken_, commonModeDigis);

  edm::Handle<reco::TrackCollection> tracksCKF;
  e.getByToken(combinatorialTracks_token_, tracksCKF);

  edm::Handle<std::vector<Trajectory> > TrajectoryCollectionCKF;
  e.getByToken(trajectories_token_, TrajectoryCollectionCKF);

  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  e.getByToken(trajTrackAsso_token_, trajTrackAssociationHandle);

  edm::Handle<edmNew::DetSetVector<SiStripCluster> > theClusters;
  e.getByToken(clusters_token_, theClusters);

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  const TrackerGeometry* tkgeom = tracker.product();

  edm::ESHandle<StripClusterParameterEstimator> stripcpe;
  es.get<TkStripCPERecord>().get("StripCPEfromTrackAngle", stripcpe);

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  es.get<SiStripQualityRcd>().get(SiStripQuality_);

  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);

  edm::Handle<DetIdCollection> fedErrorIds;
  e.getByToken(digis_token_, fedErrorIds);

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(measurementTrackerHandle);

  edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
  e.getByToken(trackerEvent_token_, measurementTrackerEvent);

  edm::ESHandle<Chi2MeasurementEstimatorBase> est;
  es.get<TrackingComponentsRecord>().get("Chi2", est);

  edm::ESHandle<Propagator> prop;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", prop);
  const Propagator& thePropagator = *prop;

  events++;

  // Tracking
  LogDebug("SiStripHitEfficiency:HitEff") << "number ckf tracks found = " << tracksCKF->size() << std::endl;
  if (!tracksCKF->empty()) {
    if (cutOnTracks_ && (tracksCKF->size() >= trackMultiplicityCut_))
      return;
    if (cutOnTracks_)
      LogDebug("SiStripHitEfficiency:HitEff")
          << "starting checking good event with < " << trackMultiplicityCut_ << " tracks" << std::endl;

    EventTrackCKF++;

    // actually should do a loop over all the tracks in the event here

    // Looping over traj-track associations to be able to get traj & track informations
    for ( const auto& trajTrack : *trajTrackAssociationHandle ) {
      // for each track, fill some variables such as number of hits and momentum

      highPurity = trajTrack.val->quality(reco::TrackBase::TrackQuality::highPurity);
      auto TMeas = trajTrack.key->measurements();

      double xloc = 0.;
      double yloc = 0.;
      double xErr = 0.;
      double yErr = 0.;
      double xglob, yglob, zglob;

      // Check whether the trajectory has some missing hits
      bool hasMissingHits = false;
      for ( const auto& tm : TMeas ) {
        auto theHit = tm.recHit();
        if (theHit->getType() == TrackingRecHit::Type::missing)
          hasMissingHits = true;
      }

      // Loop on each measurement and take it into consideration
      //--------------------------------------------------------
      for (auto itm = TMeas.cbegin(); itm != TMeas.cend(); ++itm) {
        const auto theInHit = (*itm).recHit();

        LogDebug("SiStripHitEfficiency:HitEff") << "theInHit is valid = " << theInHit->isValid() << std::endl;

        unsigned int iidd = theInHit->geographicalId().rawId();

        unsigned int TKlayers = checkLayer(iidd, tTopo);
        LogDebug("SiStripHitEfficiency:HitEff") << "TKlayer from trajectory: " << TKlayers << "  from module = " << iidd
                                                << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/"
                                                << ((iidd & 0x3) == 1) << "/" << ((iidd & 0x3) == 2) << std::endl;

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
          LogDebug("SiStripHitEfficiency:HitEff") << "skipping original TM for TOB 6 or TEC 9" << std::endl;
          continue;
        }

        // Make vector of TrajectoryAtInvalidHits to hold the trajectories
        std::vector<TrajectoryAtInvalidHit> TMs;

        // Make AnalyticalPropagator to use in TAVH constructor
        AnalyticalPropagator propagator(magField.product(), anyDirection);

        // for double sided layers check both sensors--if no hit was found on either sensor surface,
        // the trajectory measurements only have one invalid hit entry on the matched surface
        // so get the TrajectoryAtInvalidHit for both surfaces and include them in the study
        if (isDoubleSided(iidd, tTopo) && ((iidd & 0x3) == 0)) {
          // do hit eff check twice--once for each sensor
          //add a TM for each surface
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 1));
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 2));
        } else if (isDoubleSided(iidd, tTopo) && (!check2DPartner(iidd, TMeas))) {
          // if only one hit was found the trajectory measurement is on that sensor surface, and the other surface from
          // the matched layer should be added to the study as well
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 1));
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator, 2));
          LogDebug("SiStripHitEfficiency:HitEff") << " found a hit with a missing partner" << std::endl;
        } else {
          //only add one TM for the single surface and the other will be added in the next iteration
          TMs.push_back(TrajectoryAtInvalidHit(*itm, tTopo, tkgeom, propagator));
        }

        //////////////////////////////////////////////
        //Now check for tracks at TOB6 and TEC9

        // to make sure we only propagate on the last TOB5 hit check the next entry isn't also in TOB5
        // to avoid bias, make sure the TOB5 hit is valid (an invalid hit on TOB5 could only exist with a valid hit on TOB6)

        bool isValid = theInHit->isValid();
        bool isLast = (itm == (TMeas.end() - 1));
        bool isLastTOB5 = true;
        if (!isLast) {
          if (checkLayer((++itm)->recHit()->geographicalId().rawId(), tTopo) == 9)
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
          const MeasurementEstimator* estimator = est.product();
          const LayerMeasurements* theLayerMeasurements =
              new LayerMeasurements(*measurementTrackerHandle, *measurementTrackerEvent);
          const TrajectoryStateOnSurface tsosTOB5 = itm->updatedState();
          std::vector<TrajectoryMeasurement> tmp =
              theLayerMeasurements->measurements(*tob6, tsosTOB5, thePropagator, *estimator);

          if (!tmp.empty()) {
            LogDebug("SiStripHitEfficiency:HitEff") << "size of TM from propagation = " << tmp.size() << std::endl;

            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            TrajectoryMeasurement tob6TM(tmp.back());
            const auto& tob6Hit = tob6TM.recHit();

            if (tob6Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << "tob6 hit actually being added to TM vector" << std::endl;
              TMs.push_back(TrajectoryAtInvalidHit(tob6TM, tTopo, tkgeom, propagator));
            }
          }
        }

        bool isLastTEC8 = true;
        if (!isLast) {
          if (checkLayer((++itm)->recHit()->geographicalId().rawId(), tTopo) == 21)
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

          const MeasurementEstimator* estimator = est.product();
          const LayerMeasurements* theLayerMeasurements =
              new LayerMeasurements(*measurementTrackerHandle, *measurementTrackerEvent);
          const TrajectoryStateOnSurface tsosTEC9 = itm->updatedState();

          // check if track on positive or negative z
          if (!(iidd == StripSubdetector::TEC))
            LogDebug("SiStripHitEfficiency:HitEff") << "there is a problem with TEC 9 extrapolation" << std::endl;

          //cout << " tec9 id = " << iidd << " and side = " << tTopo->tecSide(iidd) << std::endl;
          std::vector<TrajectoryMeasurement> tmp;
          if (tTopo->tecSide(iidd) == 1) {
            tmp = theLayerMeasurements->measurements(*tec9neg, tsosTEC9, thePropagator, *estimator);
            //cout << "on negative side" << std::endl;
          }
          if (tTopo->tecSide(iidd) == 2) {
            tmp = theLayerMeasurements->measurements(*tec9pos, tsosTEC9, thePropagator, *estimator);
            //cout << "on positive side" << std::endl;
          }

          if (!tmp.empty()) {
            // take the last of the TMs, which is always an invalid hit
            // if no detId is available, ie detId==0, then no compatible layer was crossed
            // otherwise, use that TM for the efficiency measurement
            TrajectoryMeasurement tec9TM(tmp.back());
            const auto& tec9Hit = tec9TM.recHit();

            unsigned int tec9id = tec9Hit->geographicalId().rawId();
            LogDebug("SiStripHitEfficiency:HitEff")
                << "tec9id = " << tec9id << " is Double sided = " << isDoubleSided(tec9id, tTopo)
                << "  and 0x3 = " << (tec9id & 0x3) << std::endl;

            if (tec9Hit->geographicalId().rawId() != 0) {
              LogDebug("SiStripHitEfficiency:HitEff") << "tec9 hit actually being added to TM vector" << std::endl;
              // in tec the hit can be single or doubled sided. whenever the invalid hit at the end of vector of TMs is
              // double sided it is always on the matched surface, so we need to split it into the true sensor surfaces
              if (isDoubleSided(tec9id, tTopo)) {
                TMs.push_back(TrajectoryAtInvalidHit(tec9TM, tTopo, tkgeom, propagator, 1));
                TMs.push_back(TrajectoryAtInvalidHit(tec9TM, tTopo, tkgeom, propagator, 2));
              } else
                TMs.push_back(TrajectoryAtInvalidHit(tec9TM, tTopo, tkgeom, propagator));
            }
          }  //else std::cout << "tec9 tmp empty" << std::endl;
        }

        ////////////////////////////////////////////////////////

        // Modules Constraints

        for ( const auto& tm : TMs ) {
          // --> Get trajectory from combinatedState
          iidd = tm.monodet_id();
          LogDebug("SiStripHitEfficiency:HitEff") << "setting iidd = " << iidd << " before checking efficiency and ";

          xloc = tm.localX();
          yloc = tm.localY();

          xglob = tm.globalX();
          yglob = tm.globalY();
          zglob = tm.globalZ();
          xErr = tm.localErrorX();
          yErr = tm.localErrorY();

          TrajGlbX = 0.0;
          TrajGlbY = 0.0;
          TrajGlbZ = 0.0;

          int TrajStrip = -1;

          // reget layer from iidd here, to account for TOB 6 and TEC 9 TKlayers being off
          TKlayers = checkLayer(iidd, tTopo);

          withinAcceptance = tm.withinAcceptance() && ( ! isInBondingExclusionZone(iidd, TKlayers, yloc, yErr, tTopo) );

          if ((layers == TKlayers) || (layers == 0)) {  // Look at the layer not used to reconstruct the track
            whatlayer = TKlayers;
            LogDebug("SiStripHitEfficiency:HitEff") << "Looking at layer under study" << std::endl;
            ModIsBad = 2;
            Id = 0;
            SiStripQualBad = 0;
            run = 0;
            event = 0;
            TrajLocX = 0.0;
            TrajLocY = 0.0;
            ResXSig = 0.0;
            ClusterLocX = 0.0;
            bunchx = 0;
            commonMode = -100;

            // RPhi RecHit Efficiency

            if (!theClusters->empty()) {
              LogDebug("SiStripHitEfficiency:HitEff") << "Checking clusters with size = " << theClusters->size() << std::endl;
              std::vector<std::vector<float> >
                  VCluster_info;  //fill with X residual, X residual pull, local X, sig(X), local Y, sig(Y), StoN
              const auto idsv = theClusters->find(iidd);
              if ( idsv != theClusters->end() ) {
                //if (DEBUG)      std::cout << "the ID from the dsv = " << dsv.id() << std::endl;
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "found  (ClusterId == iidd) with ClusterId = " << idsv->id() << " and iidd = " << iidd << std::endl;
                const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(iidd)));
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
                  //cout<<" Layer "<<TKlayers<<" TrajStrip: "<<nstrips<<" "<<pitch<<" "<<TrajStrip<<endl;
                }

                for ( const auto& clus : *idsv ) {
                  StripClusterParameterEstimator::LocalValues parameters = stripcpe->localParameters(clus, *stripdet);
                  float res = (parameters.first.x() - xloc);
                  float sigma = checkConsistency(parameters, xloc, xErr);
                  // The consistency is probably more accurately measured with the Chi2MeasurementEstimator. To use it
                  // you need a TransientTrackingRecHit instead of the cluster
                  //theEstimator=       new Chi2MeasurementEstimator(30);
                  //const Chi2MeasurementEstimator *theEstimator(100);
                  //theEstimator->estimate(tm.tsos(), TransientTrackingRecHit);

                  if (TKlayers >= 11) {
                    res = parameters.first.x() - xloc / uxlden;  // radialy extrapolated x loc position at middle
                    sigma = abs(res) /
                            sqrt(parameters.second.xx() + xErr * xErr / uxlden / uxlden +
                                 yErr * yErr * xloc * xloc * uylfac * uylfac / uxlden / uxlden / uxlden / uxlden);
                  }

                  siStripClusterInfo_.setCluster(clus, idsv->id());
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
                  // TODO all but 0,1,2 can go (only used for debug prinout, after removing the rest)
                  VCluster_info.push_back(cluster_info);
                  LogDebug("SiStripHitEfficiency:HitEff") << "Have ID match. residual = " << VCluster_info.back()[0]
                                                          << "  res sigma = " << VCluster_info.back()[1] << std::endl;
                  LogDebug("SiStripHitEfficiency:HitEff")
                      << "trajectory measurement compatability estimate = " << (*itm).estimate() << std::endl;
                  LogDebug("SiStripHitEfficiency:HitEff")
                      << "hit position = " << parameters.first.x() << "  hit error = " << sqrt(parameters.second.xx())
                      << "  trajectory position = " << xloc << "  traj error = " << xErr << std::endl;
                }
              }
              float FinalResSig = 1000.0;
              float FinalCluster[7] = {1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0};
              if (! VCluster_info.empty()) {
                LogDebug("SiStripHitEfficiency:HitEff") << "found clusters > 0" << std::endl;
                if (VCluster_info.size() > 1) {
                  //get the smallest one
                  for ( const auto& res : VCluster_info ) {
                    if (abs(res[1]) < abs(FinalResSig)) {
                      FinalResSig = res[1];
                      for (unsigned int i = 0; i < res.size(); i++) {
                        LogDebug("SiStripHitEfficiency:HitEff")
                            << "filling final cluster. i = " << i << " before fill FinalCluster[i]=" << FinalCluster[i]
                            << " and res[i] =" << res[i] << std::endl;
                        FinalCluster[i] = res[i];
                        LogDebug("SiStripHitEfficiency:HitEff")
                            << "filling final cluster. i = " << i << " after fill FinalCluster[i]=" << FinalCluster[i]
                            << " and res[i] =" << res[i] << std::endl;
                      }
                    }
                    LogDebug("SiStripHitEfficiency:HitEff")
                        << "iresidual = " << res[0] << "  isigma = " << res[1]
                        << "  and FinalRes = " << FinalCluster[0] << std::endl;
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
                  << "Final residual in X = " << FinalCluster[0] << "+-" << (FinalCluster[0] / FinalResSig) << std::endl;
              LogDebug("SiStripHitEfficiency:HitEff") << "Checking location of trajectory: abs(yloc) = " << abs(yloc)
                                                      << "  abs(xloc) = " << abs(xloc) << std::endl;
              LogDebug("SiStripHitEfficiency:HitEff")
                  << "Checking location of cluster hit: yloc = " << FinalCluster[4] << "+-" << FinalCluster[5]
                  << "  xloc = " << FinalCluster[2] << "+-" << FinalCluster[3] << std::endl;
              LogDebug("SiStripHitEfficiency:HitEff") << "Final cluster signal to noise = " << FinalCluster[6] << std::endl;

              //
              // fill ntuple varibles
              //get global position from module id number iidd
              TrajGlbX = xglob;
              TrajGlbY = yglob;
              TrajGlbZ = zglob;

              Id = iidd;
              run = run_nr;
              event = ev_nr;
              bunchx = bunch_nr;
              //if ( SiStripQuality_->IsModuleBad(iidd) )
              if (SiStripQuality_->getBadApvs(iidd) != 0) {
                SiStripQualBad = 1;
                LogDebug("SiStripHitEfficiency:HitEff") << "strip is bad from SiStripQuality" << std::endl;
              } else {
                SiStripQualBad = 0;
                LogDebug("SiStripHitEfficiency:HitEff") << "strip is good from SiStripQuality" << std::endl;
              }

              //check for FED-detected errors and include those in SiStripQualBad
              for (unsigned int ii = 0; ii < fedErrorIds->size(); ii++) {
                if (iidd == (*fedErrorIds)[ii].rawId())
                  SiStripQualBad = 1;
              }

              TrajLocX = xloc;
              TrajLocY = yloc;
              ResXSig = FinalResSig;
              if (FinalResSig != FinalCluster[1])
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "Problem with best cluster selection because FinalResSig = " << FinalResSig
                    << " and FinalCluster[1] = " << FinalCluster[1] << std::endl;
              ClusterLocX = FinalCluster[2];

              // CM of APV crossed by traj
              if (addCommonMode_)
                if (commonModeDigis.isValid() && TrajStrip >= 0 && TrajStrip <= 768) {
                  const auto digiframe = commonModeDigis->find(iidd);
                  if (digiframe != commonModeDigis->end())
                    if ((unsigned)TrajStrip / 128 < digiframe->data.size())
                      commonMode = digiframe->data.at(TrajStrip / 128).adc();
                }

              LogDebug("SiStripHitEfficiency:HitEff") << "before check good" << std::endl;

              if (FinalResSig < 999.0) {  //could make requirement on track/hit consistency, but for
                //now take anything with a hit on the module
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "hit being counted as good " << FinalCluster[0] << " FinalRecHit " << iidd << "   TKlayers  "
                    << TKlayers << " xloc " << xloc << " yloc  " << yloc << " module " << iidd
                    << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/" << ((iidd & 0x3) == 1) << "/"
                    << ((iidd & 0x3) == 2) << std::endl;
                ModIsBad = 0;
              } else {
                LogDebug("SiStripHitEfficiency:HitEff")
                    << "hit being counted as bad   ######### Invalid RPhi FinalResX " << FinalCluster[0]
                    << " FinalRecHit " << iidd << "   TKlayers  " << TKlayers << " xloc " << xloc << " yloc  " << yloc
                    << " module " << iidd << "   matched/stereo/rphi = " << ((iidd & 0x3) == 0) << "/"
                    << ((iidd & 0x3) == 1) << "/" << ((iidd & 0x3) == 2) << std::endl;
                ModIsBad = 1;
                LogDebug("SiStripHitEfficiency:HitEff") << " RPhi Error " << sqrt(xErr * xErr + yErr * yErr)
                                                        << " ErrorX " << xErr << " yErr " << yErr << std::endl;
              }
              // traj->Fill(); // TODO - here is one entry for the calibtree -> histograms
              LogDebug("SiStripHitEfficiency:HitEff") << "after good location check" << std::endl;
            }
            LogDebug("SiStripHitEfficiency:HitEff") << "after list of clusters" << std::endl;
          }
          LogDebug("SiStripHitEfficiency:HitEff") << "After layers=TKLayers if" << std::endl;
        }
        LogDebug("SiStripHitEfficiency:HitEff") << "After looping over TrajAtValidHit list" << std::endl;
      }
      LogDebug("SiStripHitEfficiency:HitEff") << "end TMeasurement loop" << std::endl;
    }
    LogDebug("SiStripHitEfficiency:HitEff") << "end of trajectories loop" << std::endl;
  }
}

void SiStripHitEfficiencyWorker::endJob() {
  LogDebug("SiStripHitEfficiency:HitEff") << " Events Analysed             " << events << std::endl;
  LogDebug("SiStripHitEfficiency:HitEff") << " Number Of Tracked events    " << EventTrackCKF << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyWorker);

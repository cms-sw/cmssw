#include "RecoTracker/DebugTools/interface/GetTrackTrajInfo.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

// To convert detId to subdet/layer number.
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/Exception.h"

std::vector< GetTrackTrajInfo::Result > GetTrackTrajInfo::analyze(const edm::EventSetup& iSetup, const reco::Track& track) 
{
  // Determine the track trajectory and detLayer at each layer that the track produces a hit in.

  std::vector< GetTrackTrajInfo::Result > results;

  // Initialise Tracker geometry info (not sufficient to do this only on first call).
  edm::ESHandle<GeometricSearchTracker> tracker;
  iSetup.get<TrackerRecoGeometryRecord>().get( tracker );    

  // This is also needed to extrapolate amongst the tracker layers.
  edm::ESHandle<NavigationSchool> theSchool;
  iSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool",theSchool);
  NavigationSetter junk(*theSchool);

  // Get the magnetic field and use it to define a propagator for extrapolating the track trajectory.
  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);
  AnalyticalPropagator  propagator(&(*magField), alongMomentum);

  // This is used to check if a track is compatible with crossing a sensor.
  // Use +3.0 rather than default -3.0 here, so hit defined as inside acceptance if 
  // no more than 3*sigma outside detector edge, as opposed to more than 3*sigma inside detector edge.
  Chi2MeasurementEstimator estimator(30.,3.0);

  // Convert track to transientTrack, and hence to TSOS at its point of closest approach to beam.
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",trkTool_); // Needed for vertex fits
  reco::TransientTrack t_trk = trkTool_->build(track);
  TrajectoryStateOnSurface initTSOS = t_trk.impactPointState();
  LogDebug("GTTI")<<"TRACK TSOS POS: x="<<initTSOS.globalPosition().x()<<" y="<<initTSOS.globalPosition().y()<<" z="<<initTSOS.globalPosition().z();

  // Note if the track is going into +ve or -ve z.
  // This is only used to guess if the track is more likely to have hit a +ve rather than a -ve endcap
  // disk. Since the +ve and -ve disks are a long way apart, this approximate method is good enough.
  // More precise would be to check both possiblities and see which one (if either) the track crosses
  // using detLayer::compatible().
  bool posSide = track.eta() > 0;

  // Get hit patterns of this track
  const reco::HitPattern& hp = track.hitPattern(); 

  // Loop over info for each hit
  // N.B. Hits are sorted according to increasing distance from origin by
  // RecoTracker/TrackProducer/src/TrackProducerBase.cc
  for (int i = 0; i < hp.numberOfHits(); i++) {
    uint32_t hit = hp.getHitPattern(i);
    if (hp.trackerHitFilter(hit) && hp.validHitFilter(hit)) {
      uint32_t subDet = hp.getSubStructure(hit);
      uint32_t layer = hp.getLayer(hit);
      // subdet: PixelBarrel=1, PixelEndcap=2, TIB=3, TID=4, TOB=5, TEC=6
      LogDebug("GTTI")<<"    hit in subdet="<<subDet<<" layer="<<layer;

      // Get corresponding DetLayer object (based on code in GeometricSearchTracker::idToLayer(...)
      const DetLayer* detLayer = 0;
      if (subDet == StripSubdetector::TIB) {
        detLayer = tracker->tibLayers()[layer - 1];
      } else if (subDet == StripSubdetector::TOB) {
        detLayer = tracker->tobLayers()[layer - 1];
      } else if (subDet == StripSubdetector::TID) {
        detLayer = posSide ? tracker->posTidLayers()[layer - 1] : tracker->negTidLayers()[layer - 1];
      } else if (subDet == StripSubdetector::TEC) {
        detLayer = posSide ? tracker->posTecLayers()[layer - 1] : tracker->negTecLayers()[layer - 1];
      } else if (subDet == PixelSubdetector::PixelBarrel) {
        detLayer = tracker->pixelBarrelLayers()[layer - 1];
      } else if (subDet == PixelSubdetector::PixelEndcap) {
        detLayer = posSide ? tracker->posPixelForwardLayers()[layer - 1] : tracker->negPixelForwardLayers()[layer - 1];
      }

      // Store the results for this hit.
      Result result;
      result.detLayer = detLayer;

      // Check that the track crosses this layer, and get the track trajectory at the crossing point.
      std::pair<bool, TrajectoryStateOnSurface> layCross = detLayer->compatible(initTSOS, propagator, estimator);
      if (layCross.first) {
        LogDebug("GTTI")<<"crossed layer at "<<" x="<<layCross.second.globalPosition().x()<<" y="<<layCross.second.globalPosition().y()<<" z="<<layCross.second.globalPosition().z();

	// Find the sensor in this layer which is closest to the track trajectory.
	// And get the track trajectory at that sensor.
	const PropagationDirection along = alongMomentum;
	propagator.setPropagationDirection(along);
	std::vector< GeometricSearchDet::DetWithState > detWithState = detLayer->compatibleDets(initTSOS, propagator, estimator);
	// Check that at least one sensor was compatible with the track trajectory.
	if(detWithState.size() > 0) {
	  // Store track trajectory at this sensor.
	  result.valid    = true;
	  result.accurate = true;
	  result.detTSOS  = detWithState.front().second;
	  LogDebug("GTTI")<<"      Det in this layer compatible with TSOS: subdet="<<subDet<<" layer="<<layer;
	  LogDebug("GTTI")<<"      crossed sensor at x="<<result.detTSOS.globalPosition().x()<<" y="<<result.detTSOS.globalPosition().y()<<" z="<<result.detTSOS.globalPosition().z();

	} else {
	  // Track did not cross a sensor, so store approximate result from its intercept with the layer.
	  result.valid    = true;
          result.accurate = false;
          result.detTSOS  = layCross.second;
	  LogDebug("GTTI")<<"      WARNING: TSOS not compatible with any det in this layer, despite having a hit in it !";
	}

      } else {
	// Track trajectory did not cross layer. Pathological case.
        result.valid = false;
	LogDebug("GTTI")<<"      WARNING: track failed to cross layer, despite having a hit in hit !";
      }

      results.push_back(result);
    }
  }

  return results;
}

#include "RecoTracker/DebugTools/interface/FixTrackHitPattern.h"
#include "RecoTracker/DebugTools/interface/GetTrackTrajInfo.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

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

#include "FWCore/Utilities/interface/Exception.h"

FixTrackHitPattern::Result FixTrackHitPattern::analyze(const edm::EventSetup& iSetup, const reco::Track& track) 
{
  // Recalculate the inner and outer missing hit patterns. See header file for detailed comments.

  Result result;

  using namespace std;

  // Initialise Tracker geometry info (not sufficient to do this only on first call).
  edm::ESHandle<GeometricSearchTracker> tracker;
  iSetup.get<TrackerRecoGeometryRecord>().get( tracker );    

  // This is also needed to extrapolate amongst the tracker layers.
  edm::ESHandle<NavigationSchool> theSchool;
  iSetup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool",theSchool);
  NavigationSetter junk(*theSchool);

  // This is needed to determine which sensors are functioning.
  edm::ESHandle<MeasurementTracker> measTk;
  iSetup.get<CkfComponentsRecord>().get(measTk);
  // Don't do this. It tries getting the tracker clusters, which do not exist in AOD.
  // Hopefully not needed if one simply wants to know which sensors are active.
  // measTk->update(iEvent);

  // Get the magnetic field and use it to define a propagator for extrapolating the track trajectory.
  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);
  AnalyticalPropagator  propagator(&(*magField), alongMomentum);

  // This is used to check if a track is compatible with crossing a sensor.
  // Use +3.0 rather than default -3.0 here, so hit defined as inside acceptance if 
  // no more than 3*sigma outside detector edge, as opposed to more than 3*sigma inside detector edge.
  Chi2MeasurementEstimator estimator(30.,3.0);

  // Get the track trajectory and detLayer at each valid hit on the track.
  static GetTrackTrajInfo getTrackTrajInfo;
  vector<GetTrackTrajInfo::Result> trkTrajInfo = getTrackTrajInfo.analyze(iSetup, track);
  unsigned int nValidHits = trkTrajInfo.size();

  // Calculate the inner and outer hitPatterns on the track.
  // i.e. The list of where the track should have had hits inside its innermost valid hit 
  // and outside its outmost valid hit.

  enum InnerOuter {INNER = 1, OUTER=2};
  for (unsigned int inOut = INNER; inOut <= OUTER; inOut++) {

    // Get the track trajectory and detLayer at the inner/outermost valid hit.
    unsigned int ihit = (inOut == INNER) ? 0 : nValidHits - 1;

    // Check that information about the track trajectory was available for this hit.
    if (trkTrajInfo[ihit].valid) {
      const DetLayer* detLayer = trkTrajInfo[ihit].detLayer;
      const TrajectoryStateOnSurface& detTSOS = trkTrajInfo[ihit].detTSOS;
      const FreeTrajectoryState* detFTS = detTSOS.freeTrajectoryState();

      // When getting inner hit pattern, must propagate track inwards from innermost layer with valid hit.
      // When getting outer hit pattern, must propagate track outwards from outermost layer with valid hit.
      const PropagationDirection direc = (inOut == INNER) ? oppositeToMomentum : alongMomentum;

      // Find all layers this track is compatible with in the desired direction, starting from this layer.
      // Based on code in RecoTracker/TrackProducer/interface/TrackProducerBase.icc
      // N.B. The following call uses code in RecoTracker/TkNavigation/src/SimpleBarrelNavigableLayer::compatibleLayers() 
      // and RecoTracker/TkNavigation/src/SimpleNavigableLayer::wellInside(). 
      // These make some curious checks on the direction of the trajectory relative to its starting point,
      // so care was required above when calculating detFTS.
      vector<const DetLayer*> compLayers = detLayer->compatibleLayers(*detFTS, direc);
      LogDebug("FTHP")<<"Number of inner/outer "<<inOut<<" layers intercepted by track = "<<compLayers.size()<<endl;

      int counter = 0;
      reco::HitPattern newHitPattern;

      for(vector<const DetLayer *>::const_iterator it=compLayers.begin(); it!=compLayers.end();
	  ++it){
	if ((*it)->basicComponents().empty()) {
	  //this should never happen. but better protect for it
	  edm::LogWarning("FixTrackHitPattern")<<"a detlayer with no components: I can not figure out a DetId from this layer. please investigate.";
	  continue;
	}

	// Find the sensor in this layer which is closest to the track trajectory.
	// And get the track trajectory at that sensor.
	propagator.setPropagationDirection(direc);
	vector< GeometricSearchDet::DetWithState > detWithState = (*it)->compatibleDets(detTSOS, propagator, estimator);
	// Check that at least one sensor was compatible with the track trajectory.
	if(detWithState.size() > 0) {
	  // Check that this sensor is functional
	  DetId id = detWithState.front().first->geographicalId();
	  const MeasurementDet* measDet = measTk->idToDet(id);      
	  if(measDet->isActive()){    
	    // Hence record that the track should have produced a hit here, but did not.
	    // Store the information in a HitPattern.
	    InvalidTrackingRecHit  tmpHit(id, TrackingRecHit::missing);
	    newHitPattern.set(tmpHit, counter);      
            counter++; 
	  } else {
	    // Missing hit expected here, since sensor was not functioning.
	  }
	}
      }//end loop over compatible layers

      // Store this result.
      if (inOut == INNER) {
	result.innerHitPattern = newHitPattern;
      } else {
	result.outerHitPattern = newHitPattern;
      }

      // Print result for debugging.
      LogDebug("FTHP")<<"Number of missing hits "<<newHitPattern.numberOfHits()<<"/"<<counter<<endl;
      for (int j = 0; j < std::max(newHitPattern.numberOfHits(), counter); j++) {
	uint32_t hp = newHitPattern.getHitPattern(j);
	uint32_t subDet = newHitPattern.getSubStructure(hp);
	uint32_t layer = newHitPattern.getLayer(hp);
	uint32_t status = newHitPattern.getHitType(hp);
	LogDebug("FTHP")<<"           layer with no matched hit at counter="<<j<<" subdet="<<subDet<<" layer="<<layer<<" status="<<status<<endl;
      }

    } else {
      LogDebug("FTHP")<<"WARNING: could not calculate inner/outer hit pattern as trajectory info for inner/out hit missing"<<endl;
    }

  }

  return result;
}

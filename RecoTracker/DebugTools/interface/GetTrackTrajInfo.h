#ifndef RecoTracker_DebugTools_GetTrackTrajInfo_H
#define RecoTracker_DebugTools_GetTrackTrajInfo_H

/*
 * Determine the track trajectory and detLayer at each layer that the track produces a hit in.
 * This info can then be used to get the coordinates and momentum vector of the track at each of these
 * layers etc. 
 *
 * Call function analyze() for each track you are interested in. See comments below for that function.
 * From the "result" that it returns, you can do things such as result.detTSOS.globalPosition(),
 * to get the estimated position at which the track intercepts the layer.
 *
 * N.B. This information is obtained by extrapolating the track trajectory from its point of closest
 * approach to the beam-line. It is therefore approximate, and should not be used for hit resolution 
 * studies. 
 * If you are using RECO, you can get more precise results by refitting the track instead of using this
 * class. However, this class will work even on AOD.
 *
 * N.B. Your _cfg.py must load RecoTracker.Configuration.RecoTracker_cff to use this.
 *
 * Author: Ian Tomalin
 * Date: Oct. 2011
 */

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <vector>

class DetLayer;

class GetTrackTrajInfo {

public:

  // Used to return results.
  struct Result {
    // If this is false, the information in the struct for this hit is invalid, as the track trajectory 
    // did not cross this layer. (Should only happen in very rare cases).
    bool valid;
    // If this is false, then although the track trajectory intercepted the layer, it did not intercept 
    // a sensor inside the layer. (Can happen rarely, if the track scattered, for example).
    // If it is false, the detTSOS is evaluated at the intercept with the layer, not with the sensor,
    // so will be slightly less accurate.
    bool accurate;
    // This is the DetLayer returned by GeometricSearchTracker.
    // You can cast it into a specific type, such as BarrelDetLayer, before using it.
    const DetLayer* detLayer;
    // This is the track trajectory evaluated at the sensor. You can use it to get the coordinates
    // where the track crosses the sensor and its momentum vector at that point.
    TrajectoryStateOnSurface detTSOS;
  };

  GetTrackTrajInfo() {}
  
  ~GetTrackTrajInfo() {}

  // For each hit on the track, return the information listed in the struct Result above.
  // (See comments for struct Results).
  // There is a one-to-one correspondence between this vector and the hits returned by track::hitPattern().
  // i.e. They are ordered by the order in which the track crossed them. 
  std::vector<Result> analyze(const edm::EventSetup& iSetup, const reco::Track& track);

private:
  // Create map indicating r/z values of all layers/disks.
  void init (const edm::EventSetup& iSetup);

private:
  // Makes TransientTracks needed for vertex fitting.
  edm::ESHandle<TransientTrackBuilder> trkTool_;
};

#endif

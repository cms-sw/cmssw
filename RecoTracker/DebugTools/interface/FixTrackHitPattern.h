#ifndef RecoTracker_DebugTools_FixTrackHitPattern_H
#define RecoTracker_DebugTools_FixTrackHitPattern_H

/*
 * Recalculate the track HitPattern's, which one would usually get instead by calling 
 * Track::trackerExpectedHitsInner() and Track::trackerExpectedHitsOuter().
 * 
 * Those functions inside the Track class only define a hit to be missing if the extrapolated
 * track trajectory ALMOST CERTAINLY went through a functioning sensor in a given layer where no hit
 * was found.
 * This FixTrackHitPattern class does the same thing, except that it defines the hit to be missing if 
 * the extrapolated track trajectory MIGHT have gone through a functioning sensor in a given layer.
 *
 * If the uncertainty on the track trajectory is very small, the results should be very similar.
 * (Although those returned by class FixTrackHitPattern will be less accurate, since it extrapolates 
 * the track trajectory from its point of closest approach to the beam spot, whereas the functions inside
 * the Track class extrapolate it from from its point of closest approach to the inner/outermost valid
 * hit on the track).
 * 
 * However, if the uncertainty on the trajectory is large, then the functions inside the Track class
 * will often return nothing, as because of this large uncertainty, they can't be certain if the
 * track crosses a sensor in a given layer. In constrast, this class will continue to return useful
 * information.
 *
 * For example, tracks with hits only in TOB/TEC can have a very large extrapolation uncertainty 
 * in towards the beam-spot. In consequence, the functions in the Track class will often indicate
 * that they do not have missing hits in the Pixel or TIB/TID sensors, even though they actually do.
 *
 * It is recommended to compare the output of this class with that of the functions in the Track class.
 * Which is best will depend on your analysis.
 *
 * See the hypernews thread https://hypernews.cern.ch/HyperNews/CMS/get/recoTracking/1123.html for details.
 *
 * To use: Just call function analyze(...) with your chosen track.
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

class FixTrackHitPattern {

public:

  struct Result {
    reco::HitPattern innerHitPattern; // Info on missing hits inside innermost valid hit.
    reco::HitPattern outerHitPattern; // Info on missing hits outside outermost valid hit.
  };

  FixTrackHitPattern() {}
  
  ~FixTrackHitPattern() {}

  // Return the recalculated inner and outer HitPatterns on the track.
  Result analyze(const edm::EventSetup& iSetup, const reco::Track& track);

private:
  // Create map indicating r/z values of all layers/disks.
  void init (const edm::EventSetup& iSetup);

private:
 // Makes TransientTracks needed for vertex fitting.
  edm::ESHandle<TransientTrackBuilder> trkTool_;
};

#endif

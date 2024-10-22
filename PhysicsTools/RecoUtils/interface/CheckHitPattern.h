#ifndef PhysicsTools_RecoUtils_CheckHitPattern_H
#define PhysicsTools_RecoUtils_CheckHitPattern_H

/*
 * Determine if a track has hits in front of its assumed production point.
 * Also determine if it misses hits between its assumed production point and its innermost hit.
 *
 * FIXME: as it stands it is pretty inefficient for numerous reasons
 *        if used seriously it needs to be optimized and used properly... 
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include <utility>
#include <map>

class DetId;

class TrackerTopology;
class TrackerGeometry;

class CheckHitPattern {
public:
  struct Result {
    // Number of hits track has in front of the vertex.
    unsigned int hitsInFrontOfVert;
    // Number of missing hits between the vertex position and the innermost valid hit on the track.
    unsigned int missHitsAfterVert;
  };

  // Check if hit pattern of this track is consistent with it being produced
  // at given vertex. See comments above for "Result" struct for details of returned information.
  Result operator()(const reco::Track& track, const VertexState& vert) const;

  // Print hit pattern on track
  static void print(const reco::Track& track);

  // Create map indicating r/z values of all layers/disks.
  void init(const TrackerTopology* tTopo, const TrackerGeometry& geom, const TransientTrackBuilder& builder);

  // Return a pair<uint32, uint32> consisting of the numbers used by HitPattern to
  // identify subdetector and layer number respectively.
  typedef std::pair<uint32_t, uint32_t> DetInfo;
  static DetInfo interpretDetId(DetId detId, const TrackerTopology* tTopo);

  // Return a bool indicating if a given subdetector is in the barrel.
  static bool barrel(uint32_t subDet);

  static void print(const reco::HitPattern::HitCategory category, const reco::HitPattern& hp);

private:
  // Note if geometry info is already initialized.
  bool geomInitDone_ = false;

  // For a given subdetector & layer number, this stores the minimum and maximum
  // r (or z) values if it is barrel (or endcap) respectively.
  typedef std::map<DetInfo, std::pair<double, double> > RZrangeMap;
  RZrangeMap rangeRorZ_;

  // Makes TransientTracks needed for vertex fitting.
  const TransientTrackBuilder* trkTool_ = nullptr;
};

#endif

#ifndef PhysicsTools_RecoUtils_CheckHitPattern_H
#define PhysicsTools_RecoUtils_CheckHitPattern_H

/*
 * Determine if a track has hits in front of its assumed production point.
 * Also determine if it misses hits between its assumed production point and its innermost hit.
 */

// standard EDAnalyser include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <utility>
#include <map>

class DetId;

class TrackerTopology;

class CheckHitPattern {

public:

  struct Result {
    // Number of hits track has in front of the vertex.
    unsigned int hitsInFrontOfVert;
    // Number of missing hits between the vertex position and the innermost valid hit on the track.
    unsigned int missHitsAfterVert;
  };

  CheckHitPattern() : geomInitDone_(false) {}
  
  ~CheckHitPattern() {}

  // Check if hit pattern of this track is consistent with it being produced
  // at given vertex. See comments above for "Result" struct for details of returned information.
  // N.B. If FixHitPattern = true, then Result.missHitsAfterVert will be calculated after rederiving
  // the missing hit pattern. This rederivation is sometimes a good idea, since otherwise the
  // number of missing hits can be substantially underestimated. See comments in FixTrackHitPattern.h
  // for details.
  Result analyze(const edm::EventSetup& iSetup, 
                 const reco::Track& track, const VertexState& vert, bool fixHitPattern=true);

  // Print hit pattern on track
  void print(const reco::Track& track) const;

private:
  // Create map indicating r/z values of all layers/disks.
  void init (const edm::EventSetup& iSetup);

  // Return a pair<uint32, uint32> consisting of the numbers used by HitPattern to 
  // identify subdetector and layer number respectively.
  typedef std::pair<uint32_t, uint32_t> DetInfo;
  static DetInfo interpretDetId(DetId detId, edm::ESHandle<TrackerTopology>& tTopo);

  // Return a bool indicating if a given subdetector is in the barrel.
  static bool barrel(uint32_t subDet);

  void print(const reco::HitPattern& hp) const;

private:
  // Note if geometry info is already initialized.
  bool geomInitDone_;

  // For a given subdetector & layer number, this stores the minimum and maximum
  // r (or z) values if it is barrel (or endcap) respectively.
  typedef std::map< DetInfo, std::pair< double, double> > RZrangeMap;
  static RZrangeMap rangeRorZ_;

 // Makes TransientTracks needed for vertex fitting.
  edm::ESHandle<TransientTrackBuilder> trkTool_;
};

#endif

#ifndef PFCheckHitPattern_H
#define PFCheckHitPattern_H

// standard EDAnalyser include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#define DEBUG_CHECKHITPATTERN

class DetId;

/// \brief PFCheckHitPatter
/*!
\author Ian Tomalin, modified by Maxime Gouzevitch
\date October 2009
*/

/*
 * Determine if a track has hits in front of its assumed production point.
 * Also determine if it misses hits between its assumed production point and its innermost hit.
 */

class PFCheckHitPattern {

 public:

  PFCheckHitPattern() : geomInitDone_(false) {}
  
  ~PFCheckHitPattern() {}

  typedef std::pair <unsigned int, unsigned int> PFTrackHitInfo;
  typedef std::pair <PFTrackHitInfo, PFTrackHitInfo> PFTrackHitFullInfo;

  /// PFCheck if hit pattern of this track is consistent with it being produced
  /// at given vertex. Pair.first gives number of hits on track in front of vertex.
  /// Pair.second gives number of missing hits between vertex and innermost hit
  /// on track.

  PFTrackHitFullInfo 
    analyze(edm::ESHandle<TrackerGeometry>, edm::ESHandle<TrackerTopology>, 
	    const reco::TrackBaseRef track, 
	    const TransientVertex& vert);

  /// Print hit pattern on track
  void print(const reco::TrackBaseRef track) const;



private:
  /// Create map indicating r/z values of all layers/disks.
  void init (edm::ESHandle<TrackerGeometry>,
	     edm::ESHandle<TrackerTopology>);

  /// Return a pair<uint32, uint32> consisting of the numbers used by HitPattern to 
  /// identify subdetector and layer number respectively.
  typedef std::pair<uint32_t, uint32_t> DetInfo;

  /// Return a bool indicating if a given subdetector is in the barrel.
  static bool barrel(uint32_t subDet);

  void print(const reco::HitPattern& hp) const;

private:
  /// Note if geometry info is already initialized.
  bool geomInitDone_;

  /// For a given subdetector & layer number, this stores the minimum and maximum
  /// r (or z) values if it is barrel (or endcap) respectively.
  typedef std::map< DetInfo, std::pair< double, double> > RZrangeMap;
  static RZrangeMap rangeRorZ_;


};

#endif

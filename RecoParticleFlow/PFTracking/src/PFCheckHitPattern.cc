#include "RecoParticleFlow/PFTracking/interface/PFCheckHitPattern.h"

// To get Tracker Geometry
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

// To convert detId to subdet/layer number.
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <map>

using namespace reco;
using namespace std;

void PFCheckHitPattern::init(const TrackerTopology* tkerTopo, const TrackerGeometry* tkerGeom) {

  //
  // Note min/max radius (z) of each barrel layer (endcap disk).
  //

  geomInitDone_ = true;

  // Get Tracker geometry
  const TrackingGeometry::DetContainer& dets = tkerGeom->dets();

  // Loop over all modules in the Tracker.
  for (unsigned int i = 0; i < dets.size(); i++) {    

    // Get subdet and layer of this module
    auto detId = dets[i]->geographicalId();
    auto detInfo = DetInfo(detId.subdetId(), tkerTopo->layer(detId));
    uint32_t subDet = detInfo.first;

    // Note r (or z) of module if barrel (or endcap).
    double r_or_z;
    if (this->barrel(subDet)) {
      r_or_z = dets[i]->position().perp();
    } else {
      r_or_z = fabs(dets[i]->position().z());
    }

    // Recover min/max r/z value of this layer/disk found so far.
    double minRZ = 999.;
    double maxRZ = 0.;
    if (rangeRorZ_.find(detInfo) != rangeRorZ_.end()) {
      minRZ = rangeRorZ_[detInfo].first;
      maxRZ = rangeRorZ_[detInfo].second;
    }

    // Update with those of this module if it exceeds them.
    if (minRZ > r_or_z) minRZ = r_or_z; 
    if (maxRZ < r_or_z) maxRZ = r_or_z;     
    rangeRorZ_[detInfo] = pair<double, double>(minRZ, maxRZ);
  }

#if 0
  //def DEBUG_CHECKHITPATTERN
  RZrangeMap::const_iterator d;
  for (d = rangeRorZ_.begin(); d != rangeRorZ_.end(); d++) {
    DetInfo detInfo = d->first;
    pair<double, double> rangeRZ = d->second;
  }
#endif
}

bool PFCheckHitPattern::barrel(uint32_t subDet) {
  // Determines if given sub-detector is in the barrel.
  return (subDet == StripSubdetector::TIB || subDet == StripSubdetector::TOB ||
          subDet == PixelSubdetector::PixelBarrel); 
}


pair< PFCheckHitPattern::PFTrackHitInfo, PFCheckHitPattern::PFTrackHitInfo> 
PFCheckHitPattern::analyze(const TrackerTopology* tkerTopo, const TrackerGeometry* tkerGeom,
			   const TrackBaseRef track, const TransientVertex& vert) 
{

  // PFCheck if hit pattern of this track is consistent with it being produced
  // at given vertex. Pair.first gives number of hits on track in front of vertex.
  // Pair.second gives number of missing hits between vertex and innermost hit
  // on track.

  // Initialise geometry info if not yet done.
  if (!geomInitDone_) this->init(tkerTopo, tkerGeom);

  // Get hit patterns of this track
  const reco::HitPattern& hp = track.get()->hitPattern();

  // Count number of valid hits on track definately in front of the vertex,
  // taking into account finite depth of each layer.
  unsigned int nHitBefore = 0;
  unsigned int nHitAfter = 0;

  for (int i = 0; i < hp.numberOfHits(HitPattern::TRACK_HITS); i++) {
    uint32_t hit = hp.getHitPattern(HitPattern::TRACK_HITS, i);
    if (hp.trackerHitFilter(hit) && hp.validHitFilter(hit)) {
      uint32_t subDet = hp.getSubStructure(hit);
      uint32_t layer = hp.getLayer(hit);
      DetInfo detInfo(subDet, layer);
      double maxRZ = rangeRorZ_[detInfo].second;

      if (this->barrel(subDet)) {
        if (vert.position().perp() > maxRZ) nHitBefore++;
	else nHitAfter++;
      } else {
        if (fabs(vert.position().z()) > maxRZ) nHitBefore++;
	else nHitAfter++;
      } 
    }
  }

  // Count number of missing hits before the innermost hit on the track,
  // taking into account finite depth of each layer.
  unsigned int nMissHitAfter = 0;
  unsigned int nMissHitBefore = 0;

  for (int i = 0; i < hp.numberOfHits(HitPattern::MISSING_INNER_HITS); i++) {
    uint32_t hit = hp.getHitPattern(HitPattern::MISSING_INNER_HITS, i);
    if (reco::HitPattern::trackerHitFilter(hit)) {
      uint32_t subDet = reco::HitPattern::getSubStructure(hit);
      uint32_t layer = reco::HitPattern::getLayer(hit);
      DetInfo detInfo(subDet, layer);
      double minRZ = rangeRorZ_[detInfo].first;

      //      cout << "subDet = " << subDet << " layer = " << layer << " minRZ = " << minRZ << endl;

      if (this->barrel(subDet)) {
	if (vert.position().perp() < minRZ) nMissHitAfter++;
	else nMissHitBefore++;
      } else {
	if (fabs(vert.position().z()) < minRZ) nMissHitAfter++; 
	else nMissHitBefore++;
      } 
    }
  }


  PFTrackHitInfo trackToVertex(nHitBefore, nMissHitBefore);
  PFTrackHitInfo trackFromVertex(nHitAfter, nMissHitAfter);


  return pair< PFTrackHitInfo, PFTrackHitInfo>(trackToVertex, trackFromVertex);
}

void PFCheckHitPattern::print(const TrackBaseRef track) const {
  // Get hit patterns of this track
  const reco::HitPattern &hp = track.get()->hitPattern(); 

  cout<<"=== Hits on Track ==="<<endl;
  this->print(reco::HitPattern::TRACK_HITS, hp);
  cout<<"=== Hits before track ==="<<endl;
  this->print(reco::HitPattern::MISSING_INNER_HITS, hp);
}

void PFCheckHitPattern::print(const reco::HitPattern::HitCategory category, const reco::HitPattern& hp) const {
  for (int i = 0; i < hp.numberOfHits(category); i++) {
    uint32_t hit = hp.getHitPattern(category, i);
    if (hp.trackerHitFilter(hit)) {
      uint32_t subdet = hp.getSubStructure(hit);
      uint32_t layer = hp.getLayer(hit);
      cout<<"hit "<<i<<" subdet="<<subdet<<" layer="<<layer<<" type "<<hp.getHitType(hit)<<endl;
    }
  } 
}

#include "RecoParticleFlow/PFTracking/interface/PFCheckHitPattern.h"

// To get Tracker Geometry
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

// To convert detId to subdet/layer number.
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <map>

using namespace reco;
using namespace std;

// For a given subdetector & layer number, this static map stores the minimum and maximum
// r (or z) values if it is barrel (or endcap) respectively.
PFCheckHitPattern::RZrangeMap PFCheckHitPattern::rangeRorZ_;

void PFCheckHitPattern::init(edm::ESHandle<TrackerGeometry> tkerGeomHandle_,
			     edm::ESHandle<TrackerTopology> tTopoHand) {

  //
  // Note min/max radius (z) of each barrel layer (endcap disk).
  //

  const TrackerTopology *tTopo=tTopoHand.product();

  geomInitDone_ = true;

  // Get Tracker geometry
  const TrackingGeometry::DetContainer& dets = tkerGeomHandle_->dets();

  // Loop over all modules in the Tracker.
  for (unsigned int i = 0; i < dets.size(); i++) {    

    // Get subdet and layer of this module
    DetInfo detInfo(dets[i]->geographicalId().subdetId(), tTopo->layer(dets[i]->geographicalId()));
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
PFCheckHitPattern::analyze(edm::ESHandle<TrackerGeometry> tkerGeomHandle_, 
			   edm::ESHandle<TrackerTopology> tTopoHand,
			   const TrackBaseRef track, const TransientVertex& vert) 
{

  // PFCheck if hit pattern of this track is consistent with it being produced
  // at given vertex. Pair.first gives number of hits on track in front of vertex.
  // Pair.second gives number of missing hits between vertex and innermost hit
  // on track.

  // Initialise geometry info if not yet done.
  if (!geomInitDone_) this->init(tkerGeomHandle_,tTopoHand);

  // Get hit patterns of this track
  const reco::HitPattern& hp = track.get()->hitPattern(); 
  const reco::HitPattern& ip = track.get()->trackerExpectedHitsInner(); 

  // Count number of valid hits on track definately in front of the vertex,
  // taking into account finite depth of each layer.
  unsigned int nHitBefore = 0;
  unsigned int nHitAfter = 0;

  for (int i = 0; i < hp.numberOfHits(); i++) {
    uint32_t hit = hp.getHitPattern(i);
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

  for (int i = 0; i < ip.numberOfHits(); i++) {
    uint32_t hit = ip.getHitPattern(i);
    if (ip.trackerHitFilter(hit)) {
      uint32_t subDet = ip.getSubStructure(hit);
      uint32_t layer = ip.getLayer(hit);
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
  const reco::HitPattern& hp = track.get()->hitPattern(); 
  const reco::HitPattern& ip = track.get()->trackerExpectedHitsInner(); 

  cout<<"=== Hits on Track ==="<<endl;
  this->print(hp);
  cout<<"=== Hits before track ==="<<endl;
  this->print(ip);
}

void PFCheckHitPattern::print(const reco::HitPattern& hp) const {
  for (int i = 0; i < hp.numberOfHits(); i++) {
    uint32_t hit = hp.getHitPattern(i);
    if (hp.trackerHitFilter(hit)) {
      uint32_t subdet = hp.getSubStructure(hit);
      uint32_t layer = hp.getLayer(hit);
      cout<<"hit "<<i<<" subdet="<<subdet<<" layer="<<layer<<" type "<<hp.getHitType(hit)<<endl;
    }
  } 
}

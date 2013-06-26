#include "PhysicsTools/RecoUtils/interface/CheckHitPattern.h"
#include "RecoTracker/DebugTools/interface/FixTrackHitPattern.h"

// To get Tracker Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

// To convert detId to subdet/layer number.
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Utilities/interface/Exception.h"

// For a given subdetector & layer number, this static map stores the minimum and maximum
// r (or z) values if it is barrel (or endcap) respectively.
CheckHitPattern::RZrangeMap CheckHitPattern::rangeRorZ_;

void CheckHitPattern::init(const edm::EventSetup& iSetup) {

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //
  // Note min/max radius (z) of each barrel layer (endcap disk).
  //

  geomInitDone_ = true;

  // Get Tracker geometry
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
  const TrackingGeometry::DetContainer& dets = trackerGeometry->dets();

  // Loop over all modules in the Tracker.
  for (unsigned int i = 0; i < dets.size(); i++) {    

    // Get subdet and layer of this module
    DetInfo detInfo = this->interpretDetId(dets[i]->geographicalId(), tTopo);
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
    rangeRorZ_[detInfo] = std::pair<double, double>(minRZ, maxRZ);
  }

#ifdef DEBUG_CHECKHITPATTERN
  RZrangeMap::const_iterator d;
  for (d = rangeRorZ_.begin(); d != rangeRorZ_.end(); d++) {
    DetInfo detInfo = d->first;
    std::pair<double, double> rangeRZ = d->second;
    std::std::cout<<"CHECKHITPATTERN: Tracker subdetector type="<<detInfo.first<<" layer="<<detInfo.second
        <<" has min r (or z) ="<<rangeRZ.first<<" and max r (or z) = "<<rangeRZ.second<<std::std::endl; 
  }
#endif
}

CheckHitPattern::DetInfo CheckHitPattern::interpretDetId(DetId detId, const TrackerTopology* tTopo) {
  // Convert detId to a pair<uint32, uint32> consisting of the numbers used by HitPattern 
  // to identify subdetector and layer number respectively.
  if (detId.subdetId() == StripSubdetector::TIB) {
    return DetInfo( detId.subdetId(), tTopo->tibLayer(detId) );
  } else if (detId.subdetId() == StripSubdetector::TOB) {
    return DetInfo( detId.subdetId(), tTopo->tobLayer(detId) );
  } else if (detId.subdetId() == StripSubdetector::TID) {
    return DetInfo( detId.subdetId(), tTopo->tidWheel(detId) );
  } else if (detId.subdetId() == StripSubdetector::TEC) {
    return DetInfo( detId.subdetId(), tTopo->tecWheel(detId) );
  } else if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
    return DetInfo( detId.subdetId(), tTopo->pxbLayer(detId) );
  } else if (detId.subdetId() == PixelSubdetector::PixelEndcap) {
    return DetInfo( detId.subdetId(), tTopo->pxfDisk(detId) );
  } else {
    throw cms::Exception("NotFound","Found DetId that is not in Tracker");
  }   
}

bool CheckHitPattern::barrel(uint32_t subDet) {
  // Determines if given sub-detector is in the barrel.
  return (subDet == StripSubdetector::TIB || subDet == StripSubdetector::TOB ||
          subDet == PixelSubdetector::PixelBarrel); 
}


CheckHitPattern::Result CheckHitPattern::analyze(const edm::EventSetup& iSetup, 
			 const reco::Track& track, const VertexState& vert, bool fixHitPattern) 
{
  // Check if hit pattern of this track is consistent with it being produced
  // at given vertex. 

  // Initialise geometry info if not yet done.
  if (!geomInitDone_) this->init(iSetup);

  // Optionally set vertex position to zero for debugging.
  // VertexState vertDebug( GlobalPoint(0.,0.,0.) , GlobalError(1e-8, 0., 1e-8, 0., 0., 1e-8) );

  // Evaluate track parameters at vertex.
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",trkTool_); // Needed for vertex fits
  reco::TransientTrack t_trk = trkTool_->build(track);
  GlobalVector p3_trk = t_trk.trajectoryStateClosestToPoint(vert.position()).momentum();
  bool trkGoesInsideOut = fabs(reco::deltaPhi<const GlobalVector, const GlobalPoint>(p3_trk, vert.position())) < 0.5*M_PI;

  LogDebug("CHP")<<"TRACK: in-->out ? "<<trkGoesInsideOut<<" dxy="<<track.dxy()<<" sz="<<track.dz()<<" eta="<<track.eta()<<" barrel hits="<<track.hitPattern().numberOfValidPixelHits()<<"/"<<track.hitPattern().numberOfValidStripTIBHits()<<"/"<<track.hitPattern().numberOfValidStripTOBHits();
  LogDebug("CHP")<<"VERT: r="<<vert.position().perp()<<" z="<<vert.position().z();
  //  if (vert.position().perp() < 3.5 && fabs(vert.position().z()) < 10. && fabs(track.eta()) < 1 && fabs(track.dxy()) < 2 && fabs(track.dz()) < 2 && track.hitPattern().numberOfValidPixelHits() == 0 && track.hitPattern().numberOfValidStripTIBHits() == 0) LogDebug("CHP")<<"LOOKATTHISTRACK";
  // Get hit patterns of this track
  const reco::HitPattern& hp = track.hitPattern(); 
  reco::HitPattern        ip = track.trackerExpectedHitsInner(); 

  // Optionally fix inner hit pattern (needed if uncertainty on track trajectory is large).
  if (fixHitPattern) {
    static FixTrackHitPattern fixTrackHitPattern;
    FixTrackHitPattern::Result fixedHP = fixTrackHitPattern.analyze(iSetup, track);
    ip = fixedHP.innerHitPattern;
  }
  
  // Count number of valid hits on track definately in front of the vertex,
  // taking into account finite depth of each layer.
  unsigned int nHitBefore = 0;
  for (int i = 0; i < hp.numberOfHits(); i++) {
    uint32_t hit = hp.getHitPattern(i);
    if (hp.trackerHitFilter(hit) && hp.validHitFilter(hit)) {
      uint32_t subDet = hp.getSubStructure(hit);
      uint32_t layer = hp.getLayer(hit);
      DetInfo detInfo(subDet, layer);
      double maxRZ = rangeRorZ_[detInfo].second;

      if (this->barrel(subDet)) {
	// Be careful. If the track starts by going outside-->in, it is allowed to have hits before the vertex !
        if (vert.position().perp() > maxRZ && trkGoesInsideOut) nHitBefore++;
      } else {
        if (fabs(vert.position().z()) > maxRZ) nHitBefore++;
      } 
    }
  }

  // Count number of missing hits before the innermost hit on the track,
  // taking into account finite depth of each layer.
  unsigned int nMissHitAfter = 0;
  for (int i = 0; i < ip.numberOfHits(); i++) {
    uint32_t hit = ip.getHitPattern(i);
    //    if (ip.trackerHitFilter(hit)) {
    if (ip.trackerHitFilter(hit) && ip.type_1_HitFilter(hit)) {
      uint32_t subDet = ip.getSubStructure(hit);
      uint32_t layer = ip.getLayer(hit);
      DetInfo detInfo(subDet, layer);
      double minRZ = rangeRorZ_[detInfo].first;

      if (this->barrel(subDet)) {
	// Be careful. If the track starts by going outside-->in, then it misses hits
	// in all layers it crosses  before its innermost valid hit.
        if (vert.position().perp() < minRZ || ! trkGoesInsideOut) nMissHitAfter++;
      } else {
	if (fabs(vert.position().z()) < minRZ) nMissHitAfter++;
      } 
    }
  }
 
  Result result;
  result.hitsInFrontOfVert = nHitBefore;
  result.missHitsAfterVert = nMissHitAfter;
  return result;
}

void CheckHitPattern::print(const reco::Track& track) const {
  // Get hit patterns of this track
  const reco::HitPattern& hp = track.hitPattern(); 
  const reco::HitPattern& ip = track.trackerExpectedHitsInner(); 

  std::cout<<"=== Hits on Track ==="<<std::endl;
  this->print(hp);
  std::cout<<"=== Hits before track ==="<<std::endl;
  this->print(ip);
}

void CheckHitPattern::print(const reco::HitPattern& hp) const {
  for (int i = 0; i < hp.numberOfHits(); i++) {
    uint32_t hit = hp.getHitPattern(i);
    if (hp.trackerHitFilter(hit)) {
      uint32_t subdet = hp.getSubStructure(hit);
      uint32_t layer = hp.getLayer(hit);
      std::cout<<"hit "<<i<<" subdet="<<subdet<<" layer="<<layer<<" type "<<hp.getHitType(hit)<<std::endl;
    }
  } 
}

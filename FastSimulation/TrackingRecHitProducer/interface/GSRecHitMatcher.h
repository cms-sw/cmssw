#ifndef FastSimulation_TrackingRecHitProducer_GSRecHitMatcher_h
#define FastSimulation_TrackingRecHitProducer_GSRecHitMatcher_h

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

class GluedGeomDet;
class GeomDetUnit;
class GeomDet;

class GSRecHitMatcher {
 public:

  typedef std::pair<LocalPoint,LocalPoint>                   StripPosition; 

  GSRecHitMatcher() {}
  ~GSRecHitMatcher() {}

  SiTrackerGSMatchedRecHit2D * match( const SiTrackerGSRecHit2D *monoRH,
				      const SiTrackerGSRecHit2D *stereoRH,
				      const GluedGeomDet* gluedDet,
				      LocalVector& trackdirection) const;
  
  
  StripPosition project(const GeomDetUnit *det,
			const GluedGeomDet* glueddet,
			const StripPosition& strip,
			const LocalVector& trackdirection) const;
  
  SiTrackerGSMatchedRecHit2D * projectOnly( const SiTrackerGSRecHit2D *monoRH,
					    const GeomDet * monoDet,
					    const GluedGeomDet* gluedDet,
					    LocalVector& ldir) const;
  
};

#endif

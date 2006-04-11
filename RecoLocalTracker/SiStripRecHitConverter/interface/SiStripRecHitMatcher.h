#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITMATCH_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITRMATCH_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

class GluedGeomDet;

#include <cfloat>

class SiStripRecHitMatcher {
public:
  
  typedef  SiStripRecHit2DLocalPosCollection::const_iterator RecHitIterator;
  typedef std::vector<const SiStripRecHit2DLocalPos *>       SimpleHitCollection;
  typedef SimpleHitCollection::const_iterator                SimpleHitIterator;

  SiStripRecHitMatcher(){};

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH,
	 RecHitIterator &begin, RecHitIterator &end, 
	 const DetId &detId, 
	 const StripTopology &topol,
	 const GeomDetUnit* stripdet,
	 const GeomDetUnit * partnerstripdet) {
    return match(monoRH,begin, end, detId, topol, stripdet,partnerstripdet,LocalVector(0.,0.,0.));
  }

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH,
	 RecHitIterator &begin, RecHitIterator &end, 
	 const DetId &detId, 
	 const StripTopology &topol,
	 const GeomDetUnit* stripdet,
	 const GeomDetUnit * partnerstripdet, 
	 LocalVector trackdirection);


  /// More convenient interface with a GluedDet

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH, 
	 SimpleHitIterator begin, SimpleHitIterator end,
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection);


private:

  /// the actual implementation

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH,
	 SimpleHitIterator begin, SimpleHitIterator end,
	 const DetId &detId, 
	 const StripTopology &topol,
	 const GeomDetUnit* stripdet,
	 const GeomDetUnit * partnerstripdet,
	 LocalVector trackdirection);


};

#endif

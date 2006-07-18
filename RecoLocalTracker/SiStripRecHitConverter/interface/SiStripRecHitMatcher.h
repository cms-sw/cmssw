#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITMATCH_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITRMATCH_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GluedGeomDet;

#include <cfloat>

class SiStripRecHitMatcher {
public:
  
  typedef  SiStripRecHit2DLocalPosCollection::const_iterator RecHitIterator;
  typedef std::vector<const SiStripRecHit2DLocalPos *>       SimpleHitCollection;
  typedef SimpleHitCollection::const_iterator                SimpleHitIterator;
  typedef std::pair<LocalPoint,LocalPoint>                   StripPosition; 
  SiStripRecHitMatcher(const edm::ParameterSet& conf);
  
  const SiStripRecHit2DMatchedLocalPos& match(const SiStripRecHit2DLocalPos *monoRH, 
					      const SiStripRecHit2DLocalPos *stereoRH,
					      const GluedGeomDet* gluedDet,
					      LocalVector trackdirection);

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH,
	 RecHitIterator &begin, RecHitIterator &end, 
	 const GluedGeomDet* gluedDet) {
    return match(monoRH,begin, end, gluedDet,LocalVector(0.,0.,0.));
  }

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH,
	 RecHitIterator &begin, RecHitIterator &end, 
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection);


  /// More convenient interface with a GluedDet

    //  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
    //match( const SiStripRecHit2DLocalPos *monoRH, 
    //	 SimpleHitIterator begin, SimpleHitIterator end,
    //	 const GluedGeomDet* gluedDet,
    //	 LocalVector trackdirection);

  // project strip coordinates on Glueddet

  StripPosition project(const GeomDetUnit *det,const GluedGeomDet* glueddet,StripPosition strip,LocalVector trackdirection);

  //private:

  /// the actual implementation

  edm::OwnVector<SiStripRecHit2DMatchedLocalPos> 
  match( const SiStripRecHit2DLocalPos *monoRH,
	 SimpleHitIterator begin, SimpleHitIterator end,
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection);
  float scale_;

};

#endif

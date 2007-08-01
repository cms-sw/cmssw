#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITMATCH_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_SISTRIPRECHITMATCH_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GluedGeomDet;

#include <cfloat>


#include <boost/function.hpp>

class SiStripRecHitMatcher {
public:
  
  typedef SiStripRecHit2DCollection::DetSet::const_iterator RecHitIterator;
  typedef std::vector<const SiStripRecHit2D *>              SimpleHitCollection;
  typedef SimpleHitCollection::const_iterator               SimpleHitIterator;

  typedef boost::function<void(SiStripMatchedRecHit2D&)>    Collector;


  typedef std::pair<LocalPoint,LocalPoint>                  StripPosition; 

  SiStripRecHitMatcher(const edm::ParameterSet& conf);
  SiStripRecHitMatcher(const double theScale);
  


  SiStripMatchedRecHit2D * match(const SiStripRecHit2D *monoRH, 
				 const SiStripRecHit2D *stereoRH,
				 const GluedGeomDet* gluedDet,
				 LocalVector trackdirection) const;
  
  SiStripMatchedRecHit2D*  match(const SiStripMatchedRecHit2D *originalRH, 
				 const GluedGeomDet* gluedDet,
				 LocalVector trackdirection) const;
  
  edm::OwnVector<SiStripMatchedRecHit2D> 
  match( const SiStripRecHit2D *monoRH,
	 RecHitIterator begin, RecHitIterator end, 
	 const GluedGeomDet* gluedDet) const {
    return match(monoRH,begin, end, gluedDet,LocalVector(0.,0.,0.));
  }

  edm::OwnVector<SiStripMatchedRecHit2D> 
  match( const SiStripRecHit2D *monoRH,
	 RecHitIterator begin, RecHitIterator end, 
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection) const;

  edm::OwnVector<SiStripMatchedRecHit2D> 
  match( const SiStripRecHit2D *monoRH,
	 SimpleHitIterator begin, SimpleHitIterator end,
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection) const;



  // project strip coordinates on Glueddet

  StripPosition project(const GeomDetUnit *det,const GluedGeomDet* glueddet,StripPosition strip,LocalVector trackdirection) const;


  //private:

 
  void
  match( const SiStripRecHit2D *monoRH,
	 SimpleHitIterator begin, SimpleHitIterator end,
	 edm::OwnVector<SiStripMatchedRecHit2D> & collector, 
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection) const;


  void
  match( const SiStripRecHit2D *monoRH,
	 SimpleHitIterator begin, SimpleHitIterator end,
	 std::vector<SiStripMatchedRecHit2D*> & collector, 
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection) const;


/// the actual implementation

  void
  match( const SiStripRecHit2D *monoRH,
	 SimpleHitIterator begin, SimpleHitIterator end,
	 Collector & collector, 
	 const GluedGeomDet* gluedDet,
	 LocalVector trackdirection) const;


  float scale_;

};

#endif

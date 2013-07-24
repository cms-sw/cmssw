/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/04/24 16:54:24 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */


#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"


GEMRecHit::GEMRecHit(const GEMDetId& gemId, int bx) :  RecHit2DLocalPos(gemId),
  theGEMId(gemId), theBx(bx),theFirstStrip(99),theClusterSize(99), theLocalPosition(), theLocalError() 
{
}

GEMRecHit::GEMRecHit() :  RecHit2DLocalPos(),
  theGEMId(), theBx(99),theFirstStrip(99),theClusterSize(99), theLocalPosition(), theLocalError() 
{
}


GEMRecHit::GEMRecHit(const GEMDetId& gemId, int bx, const LocalPoint& pos) :  RecHit2DLocalPos(gemId),
  theGEMId(gemId), theBx(bx), theFirstStrip(99),theClusterSize(99), theLocalPosition(pos) 
{
  float stripResolution = 3.0 ; //cm  this sould be taken from trimmed cluster size times strip size 
                                 //    taken out from geometry service i.e. topology
  theLocalError =
    LocalError(stripResolution*stripResolution, 0., 0.); //FIXME: is it really needed?
}



// Constructor from a local position and error, wireId and digi time.
GEMRecHit::GEMRecHit(const GEMDetId& gemId,
		     int bx,
		     const LocalPoint& pos,
		     const LocalError& err) :  RecHit2DLocalPos(gemId),
  theGEMId(gemId), theBx(bx),theFirstStrip(99), theClusterSize(99), theLocalPosition(pos), theLocalError(err) 
{
}


// Constructor from a local position and error, wireId, bx and cluster size.
GEMRecHit::GEMRecHit(const GEMDetId& gemId,
		     int bx,
		     int firstStrip,
		     int clustSize,
		     const LocalPoint& pos,
		     const LocalError& err) :  RecHit2DLocalPos(gemId),
  theGEMId(gemId), theBx(bx),theFirstStrip(firstStrip), theClusterSize(clustSize), theLocalPosition(pos), theLocalError(err) 
{
}




// Destructor
GEMRecHit::~GEMRecHit()
{
}



GEMRecHit * GEMRecHit::clone() const {
  return new GEMRecHit(*this);
}


// Access to component RecHits.
// No components rechits: it returns a null vector
std::vector<const TrackingRecHit*> GEMRecHit::recHits() const {
  std::vector<const TrackingRecHit*> nullvector;
  return nullvector; 
}



// Non-const access to component RecHits.
// No components rechits: it returns a null vector
std::vector<TrackingRecHit*> GEMRecHit::recHits() {
  std::vector<TrackingRecHit*> nullvector;
  return nullvector; 
}


// Comparison operator, based on the wireId and the digi time
bool GEMRecHit::operator==(const GEMRecHit& hit) const {
  return this->geographicalId() == hit.geographicalId(); 
}


// The ostream operator
std::ostream& operator<<(std::ostream& os, const GEMRecHit& hit) {
  os << "pos: " << hit.localPosition().x() ; 
  os << " +/- " << sqrt(hit.localPositionError().xx()) ;
  return os;
}

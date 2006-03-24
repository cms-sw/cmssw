
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/01/24 14:23:25 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"

#include "FWCore/Utilities/interface/Exception.h"


using namespace DTEnums;
using namespace std;



// Constructor without components: must use setPos and Err!
DTRecHit1DPair::DTRecHit1DPair(const DTWireId& wireId,
                               const DTDigi& digi) : theDigi(digi),
                                                     theLeftHit(wireId, Left, theDigi.time()),
                                                     theRightHit(wireId, Right, theDigi.time()) {}


// Default constructor
DTRecHit1DPair::DTRecHit1DPair() : theDigi(),
                                   theLeftHit(),
                                   theRightHit() {}


// Destructor
DTRecHit1DPair::~DTRecHit1DPair(){}



DTRecHit1DPair * DTRecHit1DPair::clone() const {
  return new DTRecHit1DPair(*this);
}



// Return the 3-dimensional local position.
// The average theLeftHit/theRightHit hits position, namely the wire position
// is returned.
LocalPoint DTRecHit1DPair::localPosition() const {
  return theLeftHit.localPosition() + 
    (theRightHit.localPosition()-theLeftHit.localPosition())/2.;
}



// Return the 3-dimensional error on the local position. 
// The error is defiened as half
// the distance between theLeftHit and theRightHit pos
LocalError DTRecHit1DPair::localPositionError() const {
  return LocalError((theRightHit.localPosition().x()-
                     theLeftHit.localPosition().x())/2.,0.,0.);
}



// Access to component RecHits.
vector<const TrackingRecHit*> DTRecHit1DPair::recHits() const {
  vector<const TrackingRecHit*> result;
  result.push_back(recHit(Left));
  result.push_back(recHit(Right));
  return result;
}



// Non-const access to component RecHits.
vector<TrackingRecHit*> DTRecHit1DPair::recHits() {
  vector<TrackingRecHit*> result;
  result.push_back(const_cast<DTRecHit1D*>(recHit(Left)));
  result.push_back(const_cast<DTRecHit1D*>(recHit(Right)));
  return result;
}



// Return the detId of the Det (a DTLayer).
DetId DTRecHit1DPair::geographicalId() const { 
  return wireId().layerId();
}


  
// Comparison operator, based on the wireId and the digi
bool DTRecHit1DPair::operator==(const DTRecHit1DPair& hit) const {
  return wireId() == hit.wireId() && digi() == hit.digi();
}



// Return position in the local (layer) coordinate system for a
// certain hypothesis about the L/R cell side
LocalPoint DTRecHit1DPair::localPosition(DTCellSide lrside) const {
  return recHit(lrside)->localPosition();
}



// Return position error in the local (layer) coordinate system for a
 // certain hypothesis about the L/R cell side
 LocalError DTRecHit1DPair::localPositionError(DTCellSide lrside) const {
   return recHit(lrside)->localPositionError();
}



// Set the 3-dimensional local position for the component hit
// corresponding to the given cell side. Default value is assumed for the error.
void DTRecHit1DPair::setPosition(DTCellSide lrside, const LocalPoint& point) {
  if(lrside != undefLR) 
    recHit(lrside)->setPosition(point);
  else throw cms::Exception("DTRecHit1DPair::setPosition with undefined LR");
}



// Set the 3-dimensional local position and error for the component hit
// corresponding to the given cell side. Default value is assumed for the error.
void DTRecHit1DPair::setPositionAndError(DTCellSide lrside,
					 const LocalPoint& point, 
					 const LocalError& err) {
  if(lrside != undefLR) {
    recHit(lrside)->setPosition(point);
    recHit(lrside)->setError(err);
     }
  else throw cms::Exception("DTRecHit1DPair::setPosition with undefined LR");
}



// Return the left/right DTRecHit1D
const DTRecHit1D* DTRecHit1DPair::recHit(DTCellSide lrSide) const {
  if(lrSide == Left) {
    return const_cast<const DTRecHit1D*>(&theLeftHit);
  } else if(lrSide == Right) {
    return const_cast<const DTRecHit1D*>(&theRightHit);
  } else {
    throw cms::Exception("DTRecHit1DPair::recHit with undefined LR");
  }
}


  
// Non const access to left/right DTRecHit1D
DTRecHit1D* DTRecHit1DPair::recHit(DTCellSide lrSide) {
  if(lrSide == Left) {
    return &theLeftHit;
  } else if(lrSide == Right) {
    return &theRightHit;
  } else {
    throw cms::Exception("DTRecHit1DPair::recHit with undefined LR");
  }
}


// Ostream operator
ostream& operator<<(ostream& os, const DTRecHit1DPair& hit) {
  os << "Pos: " << hit.localPosition() ;
  return os;
}

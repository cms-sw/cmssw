
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTOccupancyPoint.h"

#include <math.h>



DTOccupancyPoint::DTOccupancyPoint() : theMean(0.),
				       theRMS(0.) {}



DTOccupancyPoint::DTOccupancyPoint(double mean, double rms) : theMean(mean),
							      theRMS(rms) {}


DTOccupancyPoint::DTOccupancyPoint(double mean, double rms, DTLayerId layerId) : theMean(mean),
										 theRMS(rms),
										 theLayerId(layerId) {}



DTOccupancyPoint::~DTOccupancyPoint(){}




double DTOccupancyPoint::mean() const {
  return theMean;
}



double DTOccupancyPoint::rms() const {
  return theRMS;
}



double DTOccupancyPoint::distance(const DTOccupancyPoint& anotherPoint) const {
  return sqrt(deltaMean(anotherPoint)*deltaMean(anotherPoint) +
	      deltaRMS(anotherPoint)* deltaRMS(anotherPoint));
}


  
double DTOccupancyPoint::deltaMean(const DTOccupancyPoint& anotherPoint) const {
  return fabs(mean() - anotherPoint.mean());
}



double DTOccupancyPoint::deltaRMS(const DTOccupancyPoint& anotherPoint) const {
  return fabs(rms() - anotherPoint.rms());
}



bool DTOccupancyPoint::operator==(const DTOccupancyPoint& other) const {
  if(theMean == other.mean() && theRMS == other.rms()) return true;
  return false;
}



bool DTOccupancyPoint::operator!=(const DTOccupancyPoint& other) const {
  if(theMean != other.mean() || theRMS != other.rms()) return true;
  return false;
}



bool DTOccupancyPoint::operator<(const DTOccupancyPoint& other) const {
  if(distance(DTOccupancyPoint()) < other.distance(DTOccupancyPoint())) return true;
  return false;
}



double computeAverageRMS(const DTOccupancyPoint& onePoint, const DTOccupancyPoint& anotherPoint) {
  double ret = (onePoint.rms() + anotherPoint.rms())/2.;
  return ret;
}



double computeMinRMS(const DTOccupancyPoint& onePoint, const DTOccupancyPoint& anotherPoint) {
  double ret = -1;
  if(onePoint.rms() > anotherPoint.rms()) {
    ret =  anotherPoint.rms();
  } else {
    ret = onePoint.rms();
  }
  return ret;
}



void DTOccupancyPoint::setLayerId(DTLayerId layerId) {
  theLayerId = layerId;
}


DTLayerId DTOccupancyPoint::layerId() const {
  return theLayerId;
}

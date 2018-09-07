/** \class MTDEtaRange
 *
 *  a class to define eta range used in MTD Navigation
 *
 *
 * \author : L. Gray - FNAL
 *
 * Modification:
 *
 */

#include "RecoMTD/Navigation/interface/MTDEtaRange.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


MTDEtaRange::MTDEtaRange() : 
   theMin(0), theMax(0) {}
   
MTDEtaRange::MTDEtaRange(float max, float min) {
 if ( max < min ) {
   edm::LogWarning ("MTDEtaRange") << "Warning MTDEtaRange:: max < min!! correcting" <<std::endl;
   float tmp(min);
   min = max;
   max = tmp;
 }
 theMax = max;
 theMin = min;
}

MTDEtaRange::MTDEtaRange(const MTDEtaRange& range) :
    theMin(range.theMin), theMax(range.theMax) {}
/// Assignment operator
MTDEtaRange& MTDEtaRange::operator=(const MTDEtaRange& range) {

  if ( this != &range ) {
    theMin = range.theMin;
    theMax = range.theMax;
  }
  return *this;
}

bool MTDEtaRange::isInside(float eta, float error) const {

  if ( (eta+error) > max() || (eta-error) < min() ) return false;
  return true;
}
/// true if this is completely inside range
bool MTDEtaRange::isInside(const MTDEtaRange& range) const {
  if ( min() > range.min() && max() < range.max() ) return true;
  return false;
}
/// true if this overlaps with range
bool MTDEtaRange::isCompatible(const MTDEtaRange& range) const {
  if ( range.min() > max() || range.max() < min() ) return false; 
  return true;
}
/// create maximum of ranges
MTDEtaRange MTDEtaRange::add(const MTDEtaRange& range) const {
  float max = ( theMax > range.theMax ) ? theMax : range.theMax;
  float min = ( theMin < range.theMin ) ? theMin : range.theMin;
  return MTDEtaRange(max,min);
}
/// create new range of size this minus range
MTDEtaRange MTDEtaRange::subtract(const MTDEtaRange& range) const {

  if ( range.isInside(*this) ) {
    edm::LogInfo ("MTDEtaRange") << "MTDEtaRange: range is inside!" << std::endl;
    return *this;
  }
  if ( !range.isCompatible(*this) ) {
    edm::LogInfo ("MTDEtaRange") << "MTDEtaRange: no overlap between ranges" << std::endl;
    return *this;
  }

  float max = isInside(range.theMin) ? range.theMin : theMax;
  float min = isInside(range.theMax) ? range.theMax : theMin;
  return MTDEtaRange(max,min);
}



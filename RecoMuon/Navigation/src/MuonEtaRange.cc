/** \class MuonEtaRange
 *
 *  a class to define eta range used in Muon Navigation
 *
 * $Date: 2006/04/24 20:00:05 $
 * $Revision: 1.2 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

#include "RecoMuon/Navigation/interface/MuonEtaRange.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>


MuonEtaRange::MuonEtaRange() : 
   theMin(0), theMax(0) {}
   
MuonEtaRange::MuonEtaRange(float max, float min) {
 if ( max < min ) {
   edm::LogWarning ("MuonEtaRange") << "Warning MuonEtaRange:: max < min!! correcting" <<std::endl;
   float tmp(min);
   min = max;
   max = tmp;
 }
 theMax = max;
 theMin = min;
}

MuonEtaRange::MuonEtaRange(const MuonEtaRange& range) :
    theMin(range.theMin), theMax(range.theMax) {}
/// Assignment operator
MuonEtaRange& MuonEtaRange::operator=(const MuonEtaRange& range) {

  if ( this != &range ) {
    theMin = range.theMin;
    theMax = range.theMax;
  }
  return *this;
}

bool MuonEtaRange::isInside(float eta, float error) const {

  if ( (eta+error) > max() || (eta-error) < min() ) return false;
  return true;
}
/// true if this is completely inside range
bool MuonEtaRange::isInside(const MuonEtaRange& range) const {
  if ( min() > range.min() && max() < range.max() ) return true;
  return false;
}
/// true if this overlaps with range
bool MuonEtaRange::isCompatible(const MuonEtaRange& range) const {
  if ( range.min() > max() || range.max() < min() ) return false; 
  return true;
}
/// create maximum of ranges
MuonEtaRange MuonEtaRange::add(const MuonEtaRange& range) const {
  float max = ( theMax > range.theMax ) ? theMax : range.theMax;
  float min = ( theMin < range.theMin ) ? theMin : range.theMin;
  return MuonEtaRange(max,min);
}
/// create new range of size this minus range
MuonEtaRange MuonEtaRange::subtract(const MuonEtaRange& range) const {

  if ( range.isInside(*this) ) {
    edm::LogInfo ("MuonEtaRange") << "MuonEtaRange: range is inside!" << std::endl;
    return *this;
  }
  if ( !range.isCompatible(*this) ) {
    edm::LogInfo ("MuonEtaRange") << "MuonEtaRange: no overlap between ranges" << std::endl;
    return *this;
  }

  float max = isInside(range.theMin) ? range.theMin : theMax;
  float min = isInside(range.theMax) ? range.theMax : theMin;
  return MuonEtaRange(max,min);
}



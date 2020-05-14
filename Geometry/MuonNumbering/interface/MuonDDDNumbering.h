#ifndef Geometry_MuonNumbering_MuonDDDNumbering_h
#define Geometry_MuonNumbering_MuonDDDNumbering_h

/** \class MuonDDDNumbering
 *
 * class to handle the conversion to MuonBaseNumber from tree of 
 * DDD GeoHistory;  
 *
 * in the xml muon constant section one has to define
 * level, super and base constants (eg. 1000,100,1) and
 * the start value of the copy numbers (0 or 1)  
 *  
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/DDCMS/interface/ExpandedNodes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

class MuonBaseNumber;
class MuonGeometryConstants;

class MuonDDDNumbering {
public:
  MuonDDDNumbering(const MuonGeometryConstants& muonConstants);
  ~MuonDDDNumbering(){};

  MuonBaseNumber geoHistoryToBaseNumber(const DDGeoHistory& history) const;
  MuonBaseNumber geoHistoryToBaseNumber(const cms::ExpandedNodes&) const;

private:
  int getInt(const std::string& s, const DDLogicalPart& part) const;

  int theLevelPart;
  int theSuperPart;
  int theBasePart;
  int theStartCopyNo;
};

#endif

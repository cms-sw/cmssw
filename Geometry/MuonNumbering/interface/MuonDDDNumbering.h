#ifndef MuonNumbering_MuonDDDNumbering_h
#define MuonNumbering_MuonDDDNumbering_h

/** \class MuonDDDNumbering
 *
 * class to handle the conversion to MuonBaseNumber from tree of 
 * DDD GeoHistory;  
 *
 * in the xml muon constant section one has to define
 * level, super and base constants (eg. 1000,100,1) and
 * the start value of the copy numbers (0 or 1)  
 *  
 *  $Date: 2006/10/12 19:54:05 $
 *  $Revision: 1.2 $
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 */

#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

class MuonBaseNumber;
class MuonDDDConstants;

class MuonDDDNumbering {
 public:

  MuonDDDNumbering( const MuonDDDConstants& muonConstants );
  ~MuonDDDNumbering(){};
  
  MuonBaseNumber geoHistoryToBaseNumber(const DDGeoHistory & history);
  
 private:

  int getInt(const std::string & s, const DDLogicalPart & part);

  int theLevelPart;
  int theSuperPart;
  int theBasePart;
  int theStartCopyNo;

};

#endif

#ifndef FastSimulation_CaloGeometryTools_CaloDirectionOperations_h
#define FastSimulation_CaloGeometryTools_CaloDirectionOperations_h


#include "Geometry/CaloTopology/interface/CaloDirection.h"

// A set of "non standard" operations on CaloDirections
// This is FastSimulation specific !
// F. Beaudette 23/10/06

class CaloDirectionOperations
{
 public:
  CaloDirectionOperations(){;}
  ~CaloDirectionOperations(){;}

  // add directions in 2D 
  static CaloDirection add2d(const CaloDirection& dir1, const CaloDirection & dir2);

  /// unsigned int -> Side conversion 
  static CaloDirection Side(unsigned i) ;
  /// Side -> unsigned conversion
  static unsigned Side(const CaloDirection& side);

  /// unsigned int -> Direction  for the neighbours 
  static unsigned neighbourDirection(const CaloDirection& side);
  /// Direction -> unsigned conversion for the neighbours
  static CaloDirection neighbourDirection(unsigned i);

  // returns the opposite side
  static CaloDirection oppositeSide(const CaloDirection& side) ;
  static unsigned oppositeDirection(unsigned iside);
};


#endif

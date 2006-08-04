#ifndef Alignment_CommonAlignmentAlgorithm_AlignableData_h
#define Alignment_CommonAlignmentAlgorithm_AlignableData_h

#include "Geometry/Surface/interface/Surface.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include<vector>

///  Helper class to store position data of an alignable;
///  Contents: position vector, rotation matrix, DetId and TypeId;
///  can be used for both absolute and relative positions/rotations


template<class T> class AlignableData 
{

public:

  /// constructor
  AlignableData(T pos, Surface::RotationType rot, 
				unsigned int id, int objid) :
    thePos(pos), theRot(rot), theObjId(objid), theId(id) {}

  /// accessors
  T pos(void) const { return thePos; }
  Surface::RotationType rot(void) const { return theRot; }
  int objId(void) const { return theObjId; }
  unsigned int id(void) const { return theId; }

private:

  // data members

  T thePos;
  Surface::RotationType theRot;
  int theObjId;
  unsigned int theId;

};

/// Absolute position/rotation 
typedef AlignableData<GlobalPoint>  AlignableAbsData;
/// relative position/rotation 
typedef AlignableData<GlobalVector> AlignableRelData;

typedef std::vector<AlignableAbsData> AlignablePositions;
typedef std::vector<AlignableRelData> AlignableShifts;

#endif


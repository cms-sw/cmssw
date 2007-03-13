#ifndef Alignment_CommonAlignmentAlgorithm_AlignableData_h
#define Alignment_CommonAlignmentAlgorithm_AlignableData_h

#include "Alignment/CommonAlignment/interface/Utilities.h"

///  Helper class to store position data of an alignable;
///  Contents: position vector, rotation matrix, DetId and TypeId;
///  can be used for both absolute and relative positions/rotations


template<class T> class AlignableData 
{

public:

  /// constructor
  AlignableData(const T& pos,
		const align::RotationType& rot, 
		unsigned int id, int objid) :
    thePos(pos), theRot(rot), theObjId(objid), theId(id) {}

  /// accessors
  const T& pos() const { return thePos; }
  const align::RotationType& rot() const { return theRot; }
  int objId() const { return theObjId; }
  unsigned int id() const { return theId; }

private:

  // data members

  T thePos;
  align::RotationType theRot;
  int theObjId;
  unsigned int theId;

};

/// Absolute position/rotation 
typedef AlignableData<align::GlobalPoint>  AlignableAbsData;
/// relative position/rotation 
typedef AlignableData<align::GlobalVector> AlignableRelData;

typedef std::vector<AlignableAbsData> AlignablePositions;
typedef std::vector<AlignableRelData> AlignableShifts;

#endif


#ifndef Alignment_CommonAlignmentAlgorithm_AlignableData_h
#define Alignment_CommonAlignmentAlgorithm_AlignableData_h

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "CondFormats/Alignment/interface/Definitions.h"

///  Helper class to store position data of an alignable;
///  Contents: position vector, rotation matrix, DetId and TypeId;
///  can be used for both absolute and relative positions/rotations


template<class T> class AlignableData 
{

public:

  /// constructor
  AlignableData(const T& pos,
		const align::RotationType& rot, 
		align::ID id, align::StructureType objid) :
    thePos(pos), theRot(rot), theObjId(objid), theId(id) {}

  /// accessors
  const T& pos() const { return thePos; }
  const align::RotationType& rot() const { return theRot; }
  align::StructureType objId() const { return theObjId; }
  align::ID id() const { return theId; }

private:

  // data members

  T thePos;
  align::RotationType theRot;
  align::StructureType theObjId;
  align::ID theId;

};

/// Absolute position/rotation 
typedef AlignableData<align::GlobalPoint>  AlignableAbsData;
/// relative position/rotation 
typedef AlignableData<align::GlobalVector> AlignableRelData;

typedef std::vector<AlignableAbsData> AlignablePositions;
typedef std::vector<AlignableRelData> AlignableShifts;

#endif


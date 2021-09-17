/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_HitCollection_h
#define CalibPPS_AlignmentRelative_HitCollection_h

#include <vector>

//----------------------------------------------------------------------------------------------------

struct Hit {
  /// sensor id
  unsigned int id;

  /// index of read-out direction (valid are: 1 or 2)
  unsigned int dirIdx;

  /// measurement position; mm
  double position;

  /// measurement position; mm
  double sigma;

  /// global z - AlignmentGeometry::z0, mm
  double z;

  Hit(unsigned int _id = 0, unsigned int _dirIdx = 0, double _pos = 0, double _sig = 0, double _z = 0)
      : id(_id), dirIdx(_dirIdx), position(_pos), sigma(_sig), z(_z) {}
};

//----------------------------------------------------------------------------------------------------

typedef std::vector<Hit> HitCollection;

//----------------------------------------------------------------------------------------------------

#endif

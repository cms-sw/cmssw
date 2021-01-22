/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_SingularMode_h
#define CalibPPS_AlignmentRelative_SingularMode_h

#include <TVectorD.h>

/**
 *\brief 
 **/
struct SingularMode {
  /// eigen value
  double val;

  /// eigen vector
  TVectorD vec;

  /// index
  int idx;
};

#endif

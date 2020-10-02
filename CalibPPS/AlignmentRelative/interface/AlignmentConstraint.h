/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_AlignmentConstraint_h
#define CalibPPS_AlignmentRelative_AlignmentConstraint_h

#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"

#include <TVectorD.h>

#include <map>
#include <string>

/**
 *\brief An alignment constraint.
 **/
struct AlignmentConstraint {
  /// constraint value
  double val;

  /// map: AlignmentAlgorithm::QuantityClass -> constraint coefficients
  std::map<unsigned int, TVectorD> coef;

  /// label of the constraint
  std::string name;
};

#endif

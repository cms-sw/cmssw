/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Alignment_RPTrackBased_AlignmentConstraint
#define Alignment_RPTrackBased_AlignmentConstraint

#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include <TVectorD.h>

#include <map>
#include <string>

/**
 *\brief An alignment constraint.
 **/
struct AlignmentConstraint
{
  /// constraint value
  double val;

  /// map: AlignmentAlgorithm::QuantityClass -> constraint coefficients
  std::map<unsigned int, TVectorD> coef;

  /// what class is the constraint for
  AlignmentTask::QuantityClass forClass;

  /// is this an extended or basic constraint
  bool extended;

  /// label of the constraint
  std::string name;
};

#endif



// File and Version Information:
//      $Id: SprClassifierEvaluator.hh,v 1.1 2007/10/29 22:10:40 narsky Exp $
//
// Description:
//      Class SprClassifierEvaluator :
//          Evaluates various quantities for trained classifiers.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprClassifierEvaluator_HH
#define _SprClassifierEvaluator_HH

class SprAbsFilter;
class SprAbsTrainedClassifier;
class SprTrainedMultiClassLearner;
class SprCoordinateMapper;

#include <vector>
#include <utility>
#include <string>


struct SprClassifierEvaluator
{
  typedef std::pair<double,double> ValueWithError;
  typedef std::pair<std::string,ValueWithError> NameAndValue;

  /*
    Computes variable importance by randomly permuting class labels
    in the data and calculating the increase in the quadratic loss
    for two-class classifiers or misid rate for the multi-class learner.
    Returns a vector of loss increases with statistical errors.
  */
  static bool variableImportance(const SprAbsFilter* data,
				 SprAbsTrainedClassifier* trained,
				 SprTrainedMultiClassLearner* mcTrained,
				 SprCoordinateMapper* mapper,
				 unsigned nPerm,
				 std::vector<NameAndValue>& lossIncrease);
};

#endif

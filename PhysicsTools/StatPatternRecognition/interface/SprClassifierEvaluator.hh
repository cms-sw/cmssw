// File and Version Information:
//      $Id: SprClassifierEvaluator.hh,v 1.2 2007/11/30 20:13:29 narsky Exp $
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

  /*
    Computes interaction between a subset of variables "vars" and all
    other variables used for this trained classifier. Fills out a
    vector of size D, where D is the total number of variables used by
    this classifier. If the variable is included in the subset, the
    intercation for this variable is set to 1.  If the input variable
    list is empty (''), this method computes interaction between each
    input variable and all other input variables.

    Interaction is defined as Correlation(F(S),F(Xd)), where

        F(S) is the classifier response at a given point integrated
	     over all variables not included in block S
	F(Xd) is the classifier response at a given point integrated
	     over all variables except variable Xd
  */
  static bool variableInteraction(const SprAbsFilter* data,
				  SprAbsTrainedClassifier* trained,
				  SprTrainedMultiClassLearner* mcTrained,
				  SprCoordinateMapper* mapper,
				  const char* vars,
				  unsigned nPoints,
				  std::vector<NameAndValue>& interaction,
				  int verbose=0);
};

#endif

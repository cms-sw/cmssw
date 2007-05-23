// File and Version Information:
//      $Id: SprAbsTrainedMultiClassLearner.hh,v 1.3 2006/11/13 19:09:38 narsky Exp $
//
// Description:
//      Class SprAbsTrainedMultiClassLearner :
//          Interface for trained multiclass methods.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprAbsTrainedMultiClassLearner_HH
#define _SprAbsTrainedMultiClassLearner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"

#include <vector>
#include <map>
#include <iostream>
#include <string>

class SprAbsFilter;


class SprAbsTrainedMultiClassLearner
{
public:
  virtual ~SprAbsTrainedMultiClassLearner() {}

  SprAbsTrainedMultiClassLearner() {}

  SprAbsTrainedMultiClassLearner(const SprAbsTrainedMultiClassLearner& other)
  {}

  /*
    Returns classifier name.
  */
  virtual std::string name() const = 0;

  /*
    Make a clone.
  */
  virtual SprAbsTrainedMultiClassLearner* clone() const = 0;

  /*
    Classifier response for a data point. 
    Computes loss values for registered classifiers; the map key
    is the class and the value is the corresponding loss.
    The returned integer is the class for which the loss is minimal.
  */
  virtual int response(const std::vector<double>& input,
		       std::map<int,double>& output) const = 0;
  int response(const SprPoint* input, std::map<int,double>& output) const {
    return this->response(p->x_);
  }

  /*
    Print out.
  */
  virtual void print(std::ostream& os) const = 0;
};

#endif

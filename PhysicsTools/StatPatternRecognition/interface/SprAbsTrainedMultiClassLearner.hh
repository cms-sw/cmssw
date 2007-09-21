// File and Version Information:
//      $Id: SprAbsTrainedMultiClassLearner.hh,v 1.4 2007/05/14 18:08:08 narsky Exp $
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
#include <iostream>
#include <string>

class SprAbsFilter;


class SprAbsTrainedMultiClassLearner
{
public:
  virtual ~SprAbsTrainedMultiClassLearner() {}

  SprAbsTrainedMultiClassLearner() {}

  SprAbsTrainedMultiClassLearner(const SprAbsTrainedMultiClassLearner& other)
    : vars_(other.vars_) {}

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
    Returns the integer label for the predicted class.
  */
  virtual int response(const std::vector<double>& input) const = 0;

  // Access to the list of variables.
  void setVars(const std::vector<std::string>& vars) { vars_ = vars; }
  void vars(std::vector<std::string>& vars) const { vars = vars_; }
  unsigned dim() const { return vars_.size(); }

  /*
    Print out.
  */
  bool store(const char* filename) const;
  virtual void print(std::ostream& os) const = 0;

protected:
  std::vector<std::string> vars_;
};

#endif

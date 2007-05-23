// File and Version Information:
//      $Id: SprAbsTrainedClassifier.hh,v 1.4 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprAbsTrainedClassifier :
//          Interface for trained classifiers.
//          The purpose of this class is to generate response of 
//          a trained classifier on validation or test data.
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
 
#ifndef _SprAbsTrainedClassifier_HH
#define _SprAbsTrainedClassifier_HH

#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <vector>
#include <iostream>
#include <string>

class SprAbsFilter;


class SprAbsTrainedClassifier
{
public:
  virtual ~SprAbsTrainedClassifier() {}

  SprAbsTrainedClassifier() : cut_() {}

  SprAbsTrainedClassifier(const SprAbsTrainedClassifier& other)
    : cut_(other.cut_) {}

  /*
    Returns classifier name.
  */
  virtual std::string name() const = 0;

  /*
    Make a clone.
  */
  virtual SprAbsTrainedClassifier* clone() const = 0;

  /*
    Classifier response for a data point. 
    Works only for problems with two categories, e.g., signal and background.

    Note: the method with SprVector should be avoided because it converts
    SprVector into an STL vector and therefore is quite slow. Use response()
    with the STL vector!
  */
  virtual double response(const std::vector<double>& v) const = 0;
  double response(const SprVector& v) const;
  double response(const SprPoint* p) const {
    return this->response(p->x_);
  }

  /*
    Classifier response. These methods set a cut on the classifier 1D output
    and accept a point if it satisfies these cuts.
  */
  virtual void setCut(const SprCut& cut) { cut_ = cut; }
  virtual void resetCut(const SprCut& cut) { cut_.clear(); }
  virtual SprCut cut() const { return cut_; }
  virtual bool accept(const std::vector<double>& v) const {
    double r = 0;
    return this->accept(v,r);
  }
  virtual bool accept(const SprVector& v) const {
    double r = 0;
    return this->accept(v,r);
  }
  bool accept(const SprPoint* p) const {
    return this->accept(p->x_);
  }
  virtual bool accept(const std::vector<double>& v, double& response) const;
  virtual bool accept(const SprVector& v, double& response) const;
  bool accept(const SprPoint* p, double& response) const {
    return this->accept(p->x_,response);
  }

  /*
    Print out.
  */
  virtual void print(std::ostream& os) const = 0;

protected:
  SprCut cut_;// cut imposed on the classifier output for accept()
};

#endif

// File and Version Information:
//      $Id: SprTrainedMultiClassLearner.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
//
// Description:
//      Class SprTrainedMultiClassLearner :
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
 
#ifndef _SprTrainedMultiClassLearner_HH
#define _SprTrainedMultiClassLearner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprMatrix.hh"

#include <vector>
#include <map>
#include <utility>
#include <iostream>
#include <string>

class SprAbsTrainedClassifier;


class SprTrainedMultiClassLearner
{
public:
  typedef double (*SprPerEventLoss)(int,double);
  typedef double (*Spr1DTransformer)(double);

  virtual ~SprTrainedMultiClassLearner() { this->destroy(); }

  SprTrainedMultiClassLearner(const SprMatrix& indicator,
			      const std::vector<int>& mapper,
			      const std::vector<std::pair<
			      const SprAbsTrainedClassifier*,bool> >& 
			      classifiers);

  SprTrainedMultiClassLearner(const SprTrainedMultiClassLearner& other);

  /*
    Returns classifier name.
  */
  std::string name() const { return "MultiClassLearner"; }

  /*
    Make a clone.
  */
  SprTrainedMultiClassLearner* clone() const {
    return new SprTrainedMultiClassLearner(*this);
  }

  /*
    Set appropriate loss and transformation.
  */
  void setLoss(SprPerEventLoss loss, Spr1DTransformer trans=0) {
    loss_ = loss; trans_ = trans;
  }

  /*
    Classifier response for a data point. 
    Computes loss values for registered classifiers; the map key
    is the class and the value is the corresponding loss.
    The returned integer is the class for which the loss is minimal.
  */
  int response(const std::vector<double>& input,
	       std::map<int,double>& output) const;

  int response(const SprPoint* input, 
	       std::map<int,double>& output) const {
    return this->response(input->x_,output);
  }

  /*
    Print out.
  */
  void print(std::ostream& os) const;
  void printIndicatorMatrix(std::ostream& os) const;

  // returns number of categories
  unsigned nClasses() const { return mapper_.size(); }

  // returns categories
  void classes(std::vector<int>& classes) const;

private:
  void destroy();

  SprMatrix indicator_;
  std::vector<int> mapper_;// maps matrix rows onto class labels
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > classifiers_;
  SprPerEventLoss loss_;
  Spr1DTransformer trans_;
};

#endif

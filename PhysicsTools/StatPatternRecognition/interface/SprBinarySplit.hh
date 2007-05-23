// File and Version Information:
//      $Id: SprBinarySplit.hh,v 1.5 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprBinarySplit :
//          Imposes consecutive binary splits on one variate in the input data.
//          The variate is given by the index d , 0<=d<dim, where
//          dim is the dimensionality of input data.
//          All points on one side of the split are considered signal,
//          and all points on the other side of the split are considered
//          background.
/*
  Note: to speed up optimization of this classifier with AdaBoost, data points
  are sorted in the corresponding dimension in the constructor. The sorted
  order is not removed by reset().
*/
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
 
#ifndef _SprBinarySplit_HH
#define _SprBinarySplit_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <string>
#include <iostream>
#include <vector>

class SprAbsTwoClassCriterion;


class SprBinarySplit : public SprAbsClassifier
{
public:
  virtual ~SprBinarySplit() {}

  SprBinarySplit(SprAbsFilter* data, 
		 const SprAbsTwoClassCriterion* crit,
		 unsigned d);


  /*
    Classifier name.
  */
  std::string name() const { return "BinarySplit"; }

  /*
    Trains classifier on data. Returns true on success, false otherwise.
  */
  bool train(int verbose=0);

  /*
    Reset this classifier to untrained state.
  */
  bool reset();

  /*
    Replace training data.
  */
  bool setData(SprAbsFilter* data);

  /*
    Prints results of training.
  */
  void print(std::ostream& os) const;

  /*
    Make a trained classifier.
  */
  SprTrainedBinarySplit* makeTrained() const {
    return new SprTrainedBinarySplit(d_,cut_);
  }

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1) {
    cls0_ = cls0; cls1_ = cls1;
    std::cout << "Classes for binary split reset to " 
	      << cls0_ << " " << cls1_ << std::endl;
    return true;
  }

private:
  void setClasses();

  const SprAbsTwoClassCriterion* crit_;
  unsigned d_;
  SprClass cls0_;
  SprClass cls1_;
  SprCut cut_;
  int nSorted_;
  std::vector<int> sorted0_;
  std::vector<int> sorted1_;
  std::vector<double> division_;

  bool sort();// sorts points in chosen dimension
};

#endif

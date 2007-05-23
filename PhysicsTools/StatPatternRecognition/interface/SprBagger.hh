//$Id: SprBagger.hh,v 1.5 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprBagger :
//          Optimizes a specified criterion by bagging, that is,
//          by drawing many bootstrap replicas and training a new
//          classifier on this replica that optimizes the said
//          criterion. A new event is then classified by the majority vote
//          of all the classifiers trained on the replicas.
// 
//          The discrete flag has no effect on the training mechanism.
//          However, it sets the corresponding flag in the trained
//          counterpart of the bagger. For details, see SprTrainedBagger.hh.
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
 
#ifndef _SprBagger_HH
#define _SprBagger_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"

#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

class SprAbsFilter;
class SprAbsTrainedClassifier;
class SprBootstrap;
class SprAbsTwoClassCriterion;
class SprAverageLoss;
class SprClass;


class SprBagger : public SprAbsClassifier
{
public:
  virtual ~SprBagger();

  SprBagger(SprAbsFilter* data);

  SprBagger(SprAbsFilter* data, 
	    unsigned cycles, 
	    bool discrete=false);

  /*
    Classifier name.
  */
  std::string name() const { return "Bagger"; }

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
  SprTrainedBagger* makeTrained() const;

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1);

  //
  // Local methods for Bagger.
  //

  //
  // modifiers
  //

  // add a trained classifier
  bool addTrained(const SprAbsTrainedClassifier* c, bool own=false);

  // set a whole set of trained classifiers with beta coefficients
  void setTrained(const std::vector<std::pair<
		  const SprAbsTrainedClassifier*,bool> >& c) {
    trained_ = c;
  }

  // add trainable
  bool addTrainable(SprAbsClassifier* c);

  // Set cycles for AdaBoost training. If 0, no training is performed.
  void setCycles(unsigned n) { cycles_ = n; }

  /*
    Validation data.
  */
  bool setValidation(const SprAbsFilter* valData, unsigned valPrint,
		     const SprAbsTwoClassCriterion* crit, 
		     SprAverageLoss* loss);

  //
  // accessors
  //

  // number of trained classifiers
  unsigned nTrained() const { return trained_.size(); }

protected:
  void setClasses();// copies two classes from the filter
  void destroy();// destroys owned trained classifiers
  bool printValidation(unsigned cycle);// misclassd frctn for validation data
  virtual bool prepareExit(bool status=true);

  const SprAbsTwoClassCriterion* crit_;
  SprClass cls0_;
  SprClass cls1_;
  unsigned cycles_;// number of cycles for training
  bool discrete_;
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > trained_;
  std::vector<SprAbsClassifier*> trainable_;
  SprBootstrap* bootstrap_;
  const SprAbsFilter* valData_;// validation data
  std::vector<double> valBeta_;// current votes for validation data
  unsigned valPrint_;// frequency of print-outs for validation data
  SprAverageLoss* loss_;// loss for validation
  bool ownLoss_;
};

#endif

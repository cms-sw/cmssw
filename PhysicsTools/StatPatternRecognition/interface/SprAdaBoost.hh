//$Id: SprAdaBoost.hh,v 1.5 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprAdaBoost :
//          Implements AdaBoost training. See corresponding section 
//          in README.
//
//          useStandard flag has no impact on the training procedure but
//          changes the output of the trainded AdaBoost. See
//          SprTrainedAdaBoost for more detail.
//
//          bagInput=true forces AdaBoost to bootstrap training events.
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
 
#ifndef _SprAdaBoost_HH
#define _SprAdaBoost_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"

#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

class SprAbsFilter;
class SprAbsTrainedClassifier;
class SprBootstrap;
class SprClass;
class SprAverageLoss;


class SprAdaBoost : public SprAbsClassifier
{
public:
  virtual ~SprAdaBoost();

  SprAdaBoost(SprAbsFilter* data);

  SprAdaBoost(SprAbsFilter* data, 
	      unsigned cycles, 
	      bool useStandard, 
	      SprTrainedAdaBoost::AdaBoostMode 
	      mode=SprTrainedAdaBoost::Discrete, 
	      bool bagInput=false);

  /*
    Classifier name.
  */
  std::string name() const { return "AdaBoost"; }

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
  SprTrainedAdaBoost* makeTrained() const;

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1);

  //
  // Local methods for AdaBoost.
  //

  // AdaBoost mode
  void setMode(SprTrainedAdaBoost::AdaBoostMode mode) { mode_ = mode; }
  SprTrainedAdaBoost::AdaBoostMode mode() const { return mode_; }

  //
  // modifiers
  //

  // add a trained classifier
  bool addTrained(const SprAbsTrainedClassifier* c, bool own=false);

  // set a whole set of trained classifiers with beta coefficients
  void setTrained(const std::vector<std::pair<
		  const SprAbsTrainedClassifier*,bool> >& c,
		  const std::vector<double>& beta) {
    assert( beta.size() == c.size() );
    trained_ = c;
    beta_ = beta;
  }

  // Add classifier that will be trained during AdaBoost optimization.
  // If a cut is specified by the user, it will be imposed on all
  // trained classfiers created from this trainable classifier.
  // For example, for the Fisher discriminant F>0 implies that an event
  // is more signal-like than background-like and vice versa.
  // For a neural net, a typical corresponding cut is at 0.5.
  // If no cut is chosen, the cut will be adjusted by AdaBoost by
  // optimizing on the minimal weighted fraction of misidentified
  // events.
  bool addTrainable(SprAbsClassifier* c, const SprCut& cut);
  bool addTrainable(SprAbsClassifier* c) {
    return this->addTrainable(c,SprCut());
  }

  // Set cycles for AdaBoost training. If 0, no training is performed.
  void setCycles(unsigned n) { cycles_ = n; }

  /*
    Classification error for validation data will be printed-out
    every valPrint training cycles. If valPrint==0, no print-outs are done.
    One can enter a pointer to the training data here if one wishes to
    print out error for the training data. For now, the print-out will go
    into std::cout. If no loss is specified, exponential will be used 
    by default.
  */
  bool setValidation(const SprAbsFilter* valData, unsigned valPrint,
		     SprAverageLoss* loss=0);

  // Store data. This method can be used to avoid recomputation of beta
  // weights if training is resumed from a saved configuration file.
  // The method stores data with weights changed due to AdaBoost training.
  bool storeData(const char* filename) const;

  // Force AdaBoost to skip event reweighting if training is resumed
  // from a data file with adjusted weights.
  void skipInitialEventReweighting(bool skip) { skipReweighting_ = skip; }

  //
  // accessors
  //

  // number of trained classifiers
  unsigned nTrained() const { return trained_.size(); }

  // epsilon access for Epsilon and Real AdaBoosts
  void setEpsilon(double eps) { assert( eps>=0 && eps<0.5 ); epsilon_ = eps; }
  double epsilon() const { return epsilon_; }

private:
  void setClasses();// copies two classes from the filter
  void destroy();// destroys owned trained classifiers
  int reweight(const SprAbsTrainedClassifier* c, double& beta, 
	       bool useInputBeta, int verbose);
  bool prepareExit(bool status=true);// adjust before exiting
  SprCut optimizeCut(const SprAbsTrainedClassifier* c, int verbose) const;
  bool printValidation(unsigned cycle);// misclassd frctn for validation data

  SprClass cls0_;
  SprClass cls1_;
  unsigned cycles_;// number of cycles for training
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > trained_;
  std::vector<std::pair<SprAbsClassifier*,SprCut> > trainable_;
  std::vector<double> beta_;// beta coefficients for trained classifiers
  double epsilon_;// epsilon for Epsilon AdaBoosts
  const SprAbsFilter* valData_;// validation data
  std::vector<double> valBeta_;// cumulative beta weights for validation data
  unsigned valPrint_;// frequency of printouts for validation data
  std::vector<double> initialDataWeights_;// data weights before training
  std::vector<double> trainedDataWeights_;// data weights after training
  bool skipReweighting_;
  bool useStandard_;
  SprTrainedAdaBoost::AdaBoostMode mode_;
  SprBootstrap* bootstrap_;
  SprAverageLoss* loss_;// loss for validation
  bool ownLoss_;
};

#endif

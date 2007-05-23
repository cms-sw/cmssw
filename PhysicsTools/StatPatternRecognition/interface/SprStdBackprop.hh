//$Id: SprStdBackprop.hh,v 1.2 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprStdBackprop :
//          Implements StdBackprop training.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprStdBackprop_HH
#define _SprStdBackprop_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRandomNumber.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

class SprAbsFilter;
class SprAbsTrainedClassifier;
class SprClass;
class SprAverageLoss;


class SprStdBackprop : public SprAbsClassifier
{
public:
  virtual ~SprStdBackprop();

  /*
    The structure must be entered as a string that shows the number of
    input, hidden and output nodes separated by colons. The number of
    hidden layers is arbitrary. The size of the input layer must be
    equal to the dimensionality of the data. The size of the output
    layer must be equal to 1. For example, if you process 10D data,
    you can enter '10:5:1' for one hidden layer or '10:6:3:1' for 2
    hidden layers. I do not have to force the user to specify the
    input and output layer since there is only one allowed choice but
    (a) I find it useful to spell out the NN structure explicitly, and
    (b) in the future more than 1 node in the output layer will be
    allowed. IN - 11.14.2006

    eta is the learning rate.
  */
  SprStdBackprop(SprAbsFilter* data);

  SprStdBackprop(SprAbsFilter* data, 
		 unsigned cycles,
		 double eta=0.1);

  SprStdBackprop(SprAbsFilter* data, 
		 const char* structure,
		 unsigned cycles,
		 double eta=0.1);

  /*
    Classifier name.
  */
  std::string name() const { return "StdBackprop"; }

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
  SprTrainedStdBackprop* makeTrained() const;

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1) {
    cls0_ = cls0; cls1_ = cls1;
    return true;
  }

  //
  // Local methods for StdBackprop.
  //

  // read SNNS configuration
  bool readSNNS(const char* netfile);

  // read SPR configuration
  bool readSPR(const char* netfile);

  // Initialize neural net with nPoints randomly selected out of data.
  // If nPoints=0, all input events are used.
  bool init(double eta, unsigned nPoints=0);

  // Set cycles for StdBackprop training. If 0, no training is performed.
  void setCycles(unsigned n) { cycles_ = n; }
  unsigned cycles() const { return cycles_; }

  // Set learning rate.
  void setLearningRate(double eta) { eta_ = eta; }
  double learningRate() const { return eta_; }

  // Allow random permutations of inputs during training.
  // By default permutations are allowed.
  void setPermute(bool permu) { allowPermu_ = permu; }

  /*
    Classification error for validation data will be printed-out every
    valPrint training cycles. If valPrint==0, no print-outs are done.
    If no loss is specified, quadratic will be used by default.
  */
  bool setValidation(const SprAbsFilter* valData, unsigned valPrint,
		     SprAverageLoss* loss=0);

private:
  friend class SprClassifierReader;

  void setClasses();// copies two classes from the filter
  bool printValidation(unsigned cycle);// misclassd frctn for validation data
  bool createNet();
  double forward(const std::vector<double>& v);
  bool backward(int cls, double output, const std::vector<double>& etaV);
  double activate(double x, SprNNDefs::ActFun f) const;
  double act_deriv(double x, SprNNDefs::ActFun f) const;
  bool doTrain(unsigned nPoints, unsigned nCycles, 
	       double eta, bool randomizeEta, int verbose);
  bool prepareExit(bool status=true);

  bool resumeReadSPR(const char* netfile, 
		     std::ifstream& file, unsigned& skipLines);
  friend class SprAdaBoostStdBackpropReader;

  std::string structure_;
  SprClass cls0_;
  SprClass cls1_;
  unsigned cycles_;// number of cycles for training
  double eta_;// learning rate
  bool configured_;
  bool initialized_;
  double initEta_;// learning rate for initialization
  unsigned initPoints_;// number of points to use for initialization
  SprRandomNumber rndm_;
  SprIntegerPermutator permu_;
  bool allowPermu_;

  int nNodes_;
  int nLinks_;
  std::vector<SprNNDefs::NodeType>   nodeType_;
  std::vector<SprNNDefs::ActFun>     nodeActFun_;
  std::vector<double>                nodeAct_;
  std::vector<double>                nodeOut_;
  std::vector<int>                   nodeNInputLinks_;
  std::vector<int>                   nodeFirstInputLink_;
  std::vector<int>                   linkSource_;
  std::vector<double>                nodeBias_;
  std::vector<double>                linkWeight_;

  SprCut cut_;// optimized cut on the NN output

  const SprAbsFilter* valData_;
  unsigned valPrint_;
  SprAverageLoss* loss_;
  bool ownLoss_;

  std::vector<double> initialDataWeights_;
};

#endif

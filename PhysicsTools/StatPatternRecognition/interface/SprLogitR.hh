// File and Version Information:
//      $Id: SprLogitR.hh,v 1.4 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprLogitR :
//          Trains logistic regression model.
//
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprLogitR_HH
#define _SprLogitR_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"

#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"

#include <string>
#include <iostream>
#include <utility>
#include <string>

class SprAbsFilter;
class SprMatrix;


class SprLogitR : public SprAbsClassifier
{
public:
  virtual ~SprLogitR() {}

  /*
    Basic constructor. 
    eps is the desired accuracy on the per-event logit response change.
    updateFactor defines how quickly logit coefficients are updated;
      reduce this factor for a slower and a more stable algorithm. 
  */
  SprLogitR(SprAbsFilter* data, double eps, double updateFactor=1); 

  // constructor with initial estimates of logit coefficients
  SprLogitR(SprAbsFilter* data, 
	    double beta0, const SprVector& beta,
	    double eps, double updateFactor=1); 

  /*
    Classifier name.
  */
  std::string name() const { return "LogitR"; }

  /*
    Trains classifier on data. Returns true on success, false otherwise.
  */
  bool train(int verbose=0);

  /*
    Reset this classifier to untrained state.
  */
  bool reset() { 
    beta0_ = beta0Supplied_;
    beta_ = betaSupplied_;
    return true; 
  }

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
  SprTrainedLogitR* makeTrained() const;

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1) {
    cls0_ = cls0; cls1_ = cls1;
    std::cout << "Classes for LogitR reset to " 
	      << cls0_ << " " << cls1_ << std::endl;
    return true;
  }

  //
  // Local methods for LogitR discriminant.
  //

  //
  // Sets accuracy on average logit probability per event.
  // Minimization is stopped when this accuracy is achieved.
  //
  void setEpsilon(double eps) { eps_ = eps; }

  //
  // Sets estimates for logit coefficients. 
  // This must be followed by a call to train().
  //
  void setBeta(double beta0, const SprVector& beta) { 
    beta0Supplied_ = beta0_;
    betaSupplied_ = beta;
  }

  //
  // accessors
  //
  double epsilon() const { return eps_; }
  double updateFactor() const { return updateFactor_; }

private:
  void setClasses();// copies two classes from the filter
  bool iterate(const SprVector& y,
	       const SprMatrix& X, 
	       const SprVector& weights, 
	       SprVector& prob, 
	       SprVector& betafit, 
	       double& eps);

  SprClass cls0_;// class indices in the data
  SprClass cls1_;
  unsigned dim_;// dimensionality of input space
  double eps_;// accuracy of minimization
  double updateFactor_;
  unsigned nIterAllowed_;// max allowed number of iterations
  double beta0_;// additive constant in front of the dot product beta*x
  SprVector beta_;// linear coefficients
  double beta0Supplied_;
  SprVector betaSupplied_;
};

#endif

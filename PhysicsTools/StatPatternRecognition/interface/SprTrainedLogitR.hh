// File and Version Information:
//      $Id: SprTrainedLogitR.hh,v 1.4 2007/02/05 21:49:45 narsky Exp $
//
// Description:
//      Class SprTrainedLogitR :
//          Returns response of a trained LogitR.
//          The returned quantity is the probability of an event at point X
//          being signal:
//
//          Logit Formula: log(p/(1-p)) = beta0 + beta*X
//          
//          p = 1/[1+exp(-(beta0+beta*X))] is the returned response.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2006              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprTrainedLogitR_HH
#define _SprTrainedLogitR_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"

#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"

#include <vector>
#include <iostream>
#include <string>


class SprTrainedLogitR : public SprAbsTrainedClassifier
{
public:
  virtual ~SprTrainedLogitR() {}

  SprTrainedLogitR(double beta0, const SprVector& beta)
    : 
    SprAbsTrainedClassifier(), 
    beta0_(beta0),
    beta_(beta),
    standard_(false)
  {}

  SprTrainedLogitR(const SprTrainedLogitR& other)
    :
    SprAbsTrainedClassifier(other),
    beta0_(other.beta0_),
    beta_(other.beta_),
    standard_(other.standard_)
  {}

  SprTrainedLogitR* clone() const {
    return new SprTrainedLogitR(*this);
  }

  // inherited methods
  std::string name() const {
    return "LogitR";
  }
  double response(const std::vector<double>& v) const;
  double response(const SprVector& v) const;
  void print(std::ostream& os) const;

  //
  // local methods
  //
  double beta(SprVector& v) const { 
    v = beta_; 
    return beta0_;
  }

  /*
    The useStandard flag lets you switch from the standard version to
    the "normalized" version. The standard version returns normal
    Logit output ranging from -infty to +infty. The normalized
    version applied logit transform 1/[1+exp(-F)] to this output.
  */
  void useStandard()   { standard_=true; }
  void useNormalized() { standard_=false; }
  bool standard() const { return standard_; }

private:
  double beta0_;// additive term
  SprVector beta_;// logit coefficients
  bool standard_;
};

#endif

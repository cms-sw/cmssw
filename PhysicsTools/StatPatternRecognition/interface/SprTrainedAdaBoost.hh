// File and Version Information:
//      $Id: SprTrainedAdaBoost.hh,v 1.2 2007/09/21 22:32:04 narsky Exp $
//
// Description:
//      Class SprTrainedAdaBoost :
//          Implements response of the trained AdaBoost.
//
//          useStandard=true flag forces AdaBoost to return the standard
//          output ranging from -infty to +infty. useStandard=false
//          forces AdaBoost to renormalize its output to [0,1].
//
//          discrete=true flag invokes the usual AdaBoost mode when the
//          response is computed as sum of discrete responses from 
//          subclassifiers multiplied by beta weights. discrete=false
//          forces AdaBoost to call response() method for each subclassifier
//          and add the returned, perhaps continuous, values multiplied
//          by beta weights.
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
 
#ifndef _SprTrainedAdaBoost_HH
#define _SprTrainedAdaBoost_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"

#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <cassert>


class SprTrainedAdaBoost : public SprAbsTrainedClassifier
{
public:
  enum AdaBoostMode { Discrete=1, Real, Epsilon };

  virtual ~SprTrainedAdaBoost() { this->destroy(); }

  SprTrainedAdaBoost(const std::vector<
		     std::pair<const SprAbsTrainedClassifier*,bool> >& 
		     trained, 
		     const std::vector<double>& beta,
		     bool useStandard,
		     AdaBoostMode mode=Discrete);

  SprTrainedAdaBoost(const SprTrainedAdaBoost& other);

  SprTrainedAdaBoost* clone() const {
    return new SprTrainedAdaBoost(*this);
  }

  /*
    Classifier name.
  */
  std::string name() const { return "AdaBoost"; }

  /*
    Classifier response for a data point. 
    Works only for problems with two categories, e.g., signal and background.
  */
  double response(const std::vector<double>& v) const;

  /*
    Generate code.
  */
  bool generateCode(std::ostream& os) const;

  // print out
  void print(std::ostream& o) const;

  /*
    Local accessors.
  */
  const SprAbsTrainedClassifier* classifier(int i) const {
    if( i>=0 && i<(int)trained_.size() )
      return trained_[i].first;
    return 0;
  }

  void classifierList(std::vector<const SprAbsTrainedClassifier*>& classifiers)
    const {
    classifiers.clear();
    classifiers.resize(trained_.size());
    for( unsigned int i=0;i<trained_.size();i++ )
      classifiers[i] = trained_[i].first;
  }

  void betaList(std::vector<double>& beta) const {
    beta = beta_;
  }

  /*
    Local modifiers.
  */

  // mode
  void setMode(AdaBoostMode mode) { mode_ = mode; }
  AdaBoostMode mode() const { return mode_; }

  // epsilon for Real AdaBoost
  void setEpsilon(double eps) { assert( eps>=0 && eps<0.5 ); epsilon_ = eps; }
  double epsilon() const { return epsilon_; }

  /*
    The useStandard flag lets you switch from the standard version of AdaBoost
    described in statistical textbooks to the "normalized" version.
    The training procedures are identical for both; the difference is in the
    output. Output of the standard AdaBoost ranges from -infty to +infty,
    and the signal/background likelihood separation is at 0. Output
    of the normalized AdaBoost ranges from 0 to 1, and the likelihoods are
    equal at 0.5. The default is the normalized AdaBoost.
  */
  void useStandard()   { standard_=true; }
  void useNormalized() { standard_=false; }
  bool standard() const { return standard_; }

private:
  void destroy();

  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > trained_;
  std::vector<double> beta_;
  AdaBoostMode mode_;
  bool standard_;
  double epsilon_;
};

#endif

// File and Version Information:
//      $Id: SprTrainedAdaBoostCombiner.hh,v 1.3 2006/11/13 19:09:40 narsky Exp $
//
// Description:
//      Class SprTrainedAdaBoostCombiner :
//          Trained AdaBoost combiner of classifiers.
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
 
#ifndef _SprTrainedAdaBoostCombiner_HH
#define _SprTrainedAdaBoostCombiner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"

#include <string>
#include <vector>
#include <iostream>


class SprTrainedAdaBoostCombiner : public SprAbsTrainedCombiner
{
public:
  virtual ~SprTrainedAdaBoostCombiner() {
    if( ownAda_ ) {
      delete ada_;
      ownAda_ = false;
    }
  }

  SprTrainedAdaBoostCombiner(const std::vector<
			     const SprAbsTrainedClassifier*>& c,
			     const SprTrainedAdaBoost* ada,
			     bool ownAda=false) 
    : 
    SprAbsTrainedCombiner(c),
    ada_(ada),
    ownAda_(ownAda)
  {
    assert( ada_ != 0 );
  }

  SprTrainedAdaBoostCombiner(const SprTrainedAdaBoostCombiner& other)
    : 
    SprAbsTrainedCombiner(other),
    ada_(other.ada_->clone()),
    ownAda_(true)
  {}

  std::string name() const { return "AdaBoostCombiner"; }

  SprTrainedAdaBoostCombiner* clone() const {
    return new SprTrainedAdaBoostCombiner(*this);
  }

  double response(const std::vector<double>& v) const;

  void print(std::ostream& os) const {
    if( ada_ != 0 ) ada_->print(os);
  }

private:
  const SprTrainedAdaBoost* ada_;
  bool ownAda_;
};

#endif

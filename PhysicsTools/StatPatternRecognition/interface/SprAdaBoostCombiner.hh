// File and Version Information:
//      $Id: SprAdaBoostCombiner.hh,v 1.4 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprAdaBoostCombiner :
//          Uses AdaBoost to combine outputs of several classifiers.
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
 
#ifndef _SprAdaBoostCombiner_HH
#define _SprAdaBoostCombiner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoostCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <iostream>
#include <vector>
#include <string>

class SprAbsFilter;
class SprClass;


class SprAdaBoostCombiner : public SprAbsCombiner
{
public:
  virtual ~SprAdaBoostCombiner() { delete ada_; }

  SprAdaBoostCombiner(SprAbsFilter* data, 
		      unsigned cycles)
    :
    SprAbsCombiner(data),
    ada_(0),
    cycles_(cycles)
  {}

  SprAdaBoostCombiner(SprAbsFilter* data, 
		      const std::vector<const SprAbsTrainedClassifier*>& c,
		      const std::vector<std::string>& cLabels,
		      unsigned cycles)
    :
    SprAbsCombiner(data,c,cLabels),
    ada_(0),
    cycles_(cycles)
  {}

  /*
    Classifier name.
  */
  std::string name() const { return "AdaBoostCombiner"; }

  bool closeClassifierList();

  SprTrainedAdaBoostCombiner* makeTrained() const;

  SprAdaBoost* ada() const { return ada_; }

  //
  // AdaBoost wrappers
  //

  bool train(int verbose=0) {
    return (ada_==0 ? false : ada_->train(verbose));
  }

  bool reset() {
    return (ada_==0 ? false : ada_->reset()); 
  }

  void print(std::ostream& os) const { 
    if( ada_ != 0 ) ada_->print(os);
  }

  bool setClasses(const SprClass& cls0, const SprClass& cls1) {
    return (ada_==0 ? false : ada_->setClasses(cls0,cls1));
  }

  bool addTrained(const SprAbsTrainedClassifier* c, bool own=false) {
    return (ada_==0 ? false : ada_->addTrained(c,own));
  }

  bool setData(SprAbsFilter* data) {
    bool status = (ada_==0 ? false : ada_->setData(data));
    if( status )
      data_ = data;
    else
      return false;
    return true;
  }

  void setTrained(const std::vector<std::pair<
		  const SprAbsTrainedClassifier*,bool> >& c,
		  const std::vector<double>& beta) {
    if( ada_ != 0 ) ada_->setTrained(c,beta);
  }

  bool addTrainable(SprAbsClassifier* c, const SprCut& cut) {
    return (ada_==0 ? false : ada_->addTrainable(c,cut));
  }

  bool addTrainable(SprAbsClassifier* c) {
    return (ada_==0 ? false : ada_->addTrainable(c));
  }

  void setCycles(unsigned n) { 
    if( ada_ != 0 ) ada_->setCycles(n);
  }

  bool setValidation(const SprAbsFilter* valData, unsigned valPrint) {
    return (ada_==0 ? false : ada_->setValidation(valData,valPrint));
  }

  unsigned nTrained() const { 
    return (ada_==0 ? false : ada_->nTrained());
  }

private:
  SprAdaBoost* ada_;
  unsigned cycles_;
};

#endif

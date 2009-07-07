// File and Version Information:
//      $Id: SprAbsCombiner.hh,v 1.1 2007/05/23 04:16:05 rpw Exp $
//
// Description:
//      Class SprAbsCombiner :
//          Interface for classifier combiners.
//          The purpose of this class is to combine trained classifiers
//          in order to construct a global classifier with presumably
//          better performance.
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
 
#ifndef _SprAbsCombiner_HH
#define _SprAbsCombiner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"

#include <vector>
#include <string>

class SprAbsTrainedClassifier;


class SprAbsCombiner : public SprAbsClassifier
{
public:
  virtual ~SprAbsCombiner() { delete features_; }

  SprAbsCombiner(SprAbsFilter* data); 

  SprAbsCombiner(SprAbsFilter* data, 
		 const std::vector<const SprAbsTrainedClassifier*>& c,
		 const std::vector<std::string>& cLabels); 

  // add trained classifier
  void addClassifier(const SprAbsTrainedClassifier* c, const char* label) { 
    classifiers_.push_back(c); 
    classifierLabels_.push_back(label);
  }

  // This method must be called after all classifiers have been added 
  // for combination.
  virtual bool closeClassifierList() = 0;

  // return trained classifier
  const SprAbsTrainedClassifier* classifier(int i) const {
    if( i<0 || i>=(int)classifiers_.size() ) return 0;
    return classifiers_[i];
  }

  // return number of trained classifiers
  unsigned nClassifiers() const { return classifiers_.size(); }

  // get features
  SprAbsFilter* features() const { return features_; }

protected:
  std::vector<const SprAbsTrainedClassifier*> classifiers_;
  std::vector<std::string> classifierLabels_;
  SprAbsFilter* features_;

  // makes data for the combiner
  bool makeFeatures();
};

#endif

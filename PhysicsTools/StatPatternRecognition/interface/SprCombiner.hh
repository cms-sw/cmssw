// File and Version Information:
//      $Id: SprCombiner.hh,v 1.1 2007/09/21 22:32:01 narsky Exp $
//
// Description:
//      Class SprCombiner :
//          Interface for classifier combiners.
//          This class combines trained classifiers
//          and constructs a global classifier in the space of their output.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005,2007         California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprCombiner_HH
#define _SprCombiner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <cassert>
#include <iostream>

class SprAbsFilter;
class SprEmptyFilter;
class SprAbsTrainedClassifier;
class SprCoordinateMapper;


class SprCombiner : public SprAbsClassifier
{
public:
  typedef std::map<std::string,SprCut> SprAllowedStringMap;

  virtual ~SprCombiner();

  SprCombiner(SprAbsFilter* data); 

  // set trainable classifier
  void setTrainable(SprAbsClassifier* c) {
    assert( c != 0 );
    trainable_ = c;
  }

  // add trained classifier
  bool addTrained(const SprAbsTrainedClassifier* c, 
		  const char* label,
		  const SprAllowedStringMap& stringMap,
		  double defaultValue,
		  bool own=false);

  /*
    Classifier name.
  */
  std::string name() const { return "Combiner"; }

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
    Trained combiner.
  */
  SprTrainedCombiner* makeTrained() const;

  /*
    Choose two classes.
  */
  bool setClasses(const SprClass& cls0, const SprClass& cls1);

  // This method must be called after all classifiers
  // have been added for combination.
  bool closeClassifierList();

  // return trained classifier
  const SprAbsTrainedClassifier* classifier(int i) const {
    if( i<0 || i>=(int)trained_.size() ) return 0;
    return trained_[i].first;
  }

  // return number of trained classifiers
  unsigned nClassifiers() const { return trained_.size(); }

  // get features
  SprEmptyFilter* features() const { return features_; }

protected:
  typedef SprTrainedCombiner::SprAllowedIndexMap LocalIndexMap;

  // makes data for the combiner
  bool makeFeatures();

  SprAbsClassifier* trainable_;
  SprEmptyFilter* features_;
  std::vector<std::pair<const SprAbsTrainedClassifier*,bool> > trained_;
  std::vector<std::string> labels_;
  // mapping from user contraints to trained classifier vars
  std::vector<LocalIndexMap> constraints_;
  // mapping from trained classifier vars to input data vars
  std::vector<SprCoordinateMapper*> inputDataMappers_;
  std::vector<double> defaultValues_;
};

#endif

// File and Version Information:
//      $Id: SprAbsClassifier.hh,v 1.4 2007/02/05 21:49:44 narsky Exp $
//
// Description:
//      Class SprAbsClassifier :
//          Interface for untrained classifiers.
//          The purpose of this class is to train a classifier on data.
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
 
#ifndef _SprAbsClassifier_HH
#define _SprAbsClassifier_HH

#include <string>
#include <iostream>
#include <cassert>

class SprAbsFilter;
class SprAbsTrainedClassifier;
class SprClass;


class SprAbsClassifier
{
public:
  virtual ~SprAbsClassifier() {}

  SprAbsClassifier(SprAbsFilter* data) : data_(data) 
  {
    assert( data_ != 0 );
  }

  /*
    Classifier name.
  */
  virtual std::string name() const = 0;

  /*
    Trains classifier on data. Returns true on success, false otherwise.
  */
  virtual bool train(int verbose=0) = 0;

  /*
    Reset this classifier to untrained state.
  */
  virtual bool reset() = 0;

  /*
    Replace training data.
  */
  virtual bool setData(SprAbsFilter* data) = 0;

  /*
    Prints results of training.
  */
  virtual void print(std::ostream& os) const = 0;

  /*
    Store training results in a file.
  */
  virtual bool store(const char* filename) const;

  /*
    Make a trained classifier.
  */
  virtual SprAbsTrainedClassifier* makeTrained() const = 0;

  /*
    Choose two classes.
  */
  virtual bool setClasses(const SprClass& cls0, const SprClass& cls1) = 0;

protected:
  SprAbsFilter* data_;// non-const filter to allow adjustment of weights
};

#endif

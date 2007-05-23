// File and Version Information:
//      $Id: SprAbsMultiClassLearner.hh,v 1.3 2006/11/13 19:09:38 narsky Exp $
//
// Description:
//      Class SprAbsMultiClassLearner :
//          Interface for multiclass methods.
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
 
#ifndef _SprAbsMultiClassLearner_HH
#define _SprAbsMultiClassLearner_HH

#include <iostream>
#include <cassert>

class SprAbsFilter;
class SprAbsTrainedMultiClassLearner;


class SprAbsMultiClassLearner
{
public:
  virtual ~SprAbsMultiClassLearner() {}

  SprAbsMultiClassLearner(SprAbsFilter* data) : data_(data) 
  {
    assert( data_ != 0 );
  }

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
  virtual SprAbsTrainedMultiClassLearner* makeTrained() const = 0;

protected:
  SprAbsFilter* data_;// non-const filter to allow adjustment of weights
};

#endif

//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprAverageLoss.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprAverageLoss :
//         Computes average loss for a data sample. Implementations 
//         of the loss() method can assume various conventions 
//         for the true class and value predicted by a classifier.
//         For fast performance, no explicit checks will be made 
//         to enforce this convention; checking the range of the input
//         parameters for the update() method is therefore responsibility
//         of the user.
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
 
#ifndef _SprAverageLoss_HH
#define _SprAverageLoss_HH

#include <cassert>


class SprAverageLoss
{
public:
  typedef double (*SprPerEventLoss)(int,double);
  typedef double (*Spr1DTransformer)(double);

  virtual ~SprAverageLoss() {}

  SprAverageLoss(SprPerEventLoss loss, Spr1DTransformer trans=0) 
    : loss_(loss), trans_(trans), value_(0), weight_(0) 
  {
    assert( loss != 0 );
  }

  inline void update(int cls, double predicted, double weight);

  void reset() { value_ = 0; weight_ = 0; }

  double value() const { return value_; }
  double weight() const { return weight_; }

protected:
  SprPerEventLoss loss_;
  Spr1DTransformer trans_;
  double value_;
  double weight_;
};


inline void SprAverageLoss::update(int cls, double predicted, double weight) 
{
  double f = ( trans_==0 ? predicted : trans_(predicted) );
  value_ = weight_*value_ + weight*loss_(cls,f);
  weight_ += weight;
  assert( weight_ > 0 );
  value_ /= weight_;
}

#endif


//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprPoint.hh,v 1.2 2007/09/21 22:32:02 narsky Exp $
//
// Description:
//      Class SprPoint :
//          point coordinates and category
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
 
#ifndef _SprPoint_HH
#define _SprPoint_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <vector>
#include <iostream>


struct SprPoint
{
  ~SprPoint() {}

  SprPoint() : index_(0), class_(0), x_() {}

  SprPoint(unsigned index, int cls, const std::vector<double>& v)
    :
    index_(index),
    class_(cls),
    x_(v)
  {}

  SprPoint(const SprPoint& other) 
    :
    index_(other.index_),
    class_(other.class_),
    x_(other.x_)
  {}

  // methods
  double operator[](int i) const { return x_[i]; }
  inline double at(int i) const;
  unsigned dim() const { return x_.size(); }
  bool empty() const { return x_.empty(); }
  bool operator<(const SprPoint& other) const {
    return (index_ < other.index_);
  }
  bool operator==(const SprPoint& other) const {
    return !(this->operator<(other) || other.operator<(*this));
  }
  bool operator!=(const SprPoint& other) const {
    return !this->operator==(other);
  }
  SprPoint& operator=(const SprPoint& other) {
    index_ = other.index_;
    class_ = other.class_;
    x_ = other.x_;
    return *this;
  }
  bool index_eq(unsigned index) const {
    return (index_ == index);
  }
  bool class_eq(SprClass cls) const {
    return (class_ == cls);
  }

  // data
  unsigned index_;
  int class_;
  std::vector<double> x_;
};

inline double SprPoint::at(int i) const 
{
  if( i>=0 && i<(int)x_.size() ) return x_[i];
  std::cerr << "Index out of range for vector " << i << " " 
	    << x_.size() << std::endl;
  return 0;
}

#endif

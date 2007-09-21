//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprBoxFilter.hh,v 1.4 2007/05/17 23:31:37 narsky Exp $
//
// Description:
//      Class SprBoxFilter :
//         Imposes rectangular cuts on SprData.
//         Each cut is a collection of pair<double,double>(min,max),
//         which specify allowed regions.
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
 
#ifndef _SprBoxFilter_HH
#define _SprBoxFilter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <vector>

class SprPoint;


class SprBoxFilter : public SprAbsFilter
{
public:
  virtual ~SprBoxFilter() {}

  SprBoxFilter(const SprData* data) 
    : SprAbsFilter(data), box_() {}

  SprBoxFilter(const SprBoxFilter& filter) 
    : SprAbsFilter(filter), box_(filter.box_) {}

  SprBoxFilter(const SprAbsFilter* filter)
    : SprAbsFilter(*filter), box_() {}

  SprBoxFilter& operator=(const SprBoxFilter& other) {
    box_ = other.box_;
    return *this;
  }

  // accept or reject a point
  bool pass(const SprPoint* p) const;

  // specific reset
  bool reset() {
    box_.clear();
    return true;
  }

  // define box
  bool setBox(const SprBox& box) {
    box_ = box;
    return true;
  }

  // set cut in a specific dimension
  bool setRange(int d, const SprInterval& range);

  // set a full set of cuts on all dims
  bool setBox(const std::vector<SprInterval>& box);

  // access to box
  void box(SprBox& box) const { box = box_; }
  SprInterval range(int d) const;

private:
  SprBox box_;
};

#endif


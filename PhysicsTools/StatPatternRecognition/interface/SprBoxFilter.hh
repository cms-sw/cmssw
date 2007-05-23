//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprBoxFilter.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
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
    : SprAbsFilter(data), cuts_() {}

  SprBoxFilter(const SprBoxFilter& filter) 
    : SprAbsFilter(filter), cuts_(filter.cuts_) {}

  SprBoxFilter(const SprAbsFilter* filter)
    : SprAbsFilter(*filter), cuts_() {}

  // define cuts
  bool setCut(const SprGrid& cuts) { 
    cuts_ = cuts; 
    return true;
  }
  bool setCut(const std::vector<SprCut>& cuts);
  bool resetCut() {
    cuts_.clear();
    return true;
  }

  // accept or reject a point
  bool pass(const SprPoint* p) const;

  //
  // local methods
  //

  // set cut in a specific dimension
  bool setCut(int i, const SprCut& cut);

  // set cut on a variable
  bool setCut(const char* var, const SprCut& cut);

private:
  SprGrid cuts_;
};

#endif


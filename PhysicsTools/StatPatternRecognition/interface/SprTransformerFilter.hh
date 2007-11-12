//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprTransformerFilter.hh,v 1.1 2007/11/07 00:56:14 narsky Exp $
//
// Description:
//      Class SprTransformerFilter :
//         Applies a transformation to input variables.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprTransformerFilter_HH
#define _SprTransformerFilter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"

class SprPoint;
class SprData;
class SprAbsVarTransformer;


class SprTransformerFilter : public SprAbsFilter
{
public:
  virtual ~SprTransformerFilter() {}

  SprTransformerFilter(const SprData* data, 
		       bool ownData=false) 
    : SprAbsFilter(data,ownData) {}

  SprTransformerFilter(const SprTransformerFilter& filter) 
    : SprAbsFilter(filter) {}

  SprTransformerFilter(const SprAbsFilter* filter)
    : SprAbsFilter(*filter) {}

  // specific reset
  bool reset() { return true; }

  // accept or reject a point
  bool pass(const SprPoint* p) const { return true; }

  // Apply transformation.
  // If replaceOriginalData=false, this doubles the space taken by the
  //   dataset in memory.
  // If replaceOriginalData=true, this replaces the original data and the
  //   filter has a permanent effect.
  bool transform(const SprAbsVarTransformer* trans,
		 bool replaceOriginalData=false);
};

#endif


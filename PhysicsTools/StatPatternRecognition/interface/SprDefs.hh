//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprDefs.hh,v 1.8 2007/08/30 17:54:38 narsky Exp $
//
// Description:
//      Class SprDefs :
//         Collection of various definitions.
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
 
#ifndef _SprDefs_HH
#define _SprDefs_HH

#include <string>
#include <vector>
#include <map>
#include <utility>

class SprAbsClassifier;

/*
  SPR version.
*/
const std::string SprVersion="SPR-07-00-00";

// 1D interval
typedef std::pair<double,double> SprInterval;

/*
  SprCut represents a cut on one dimension. To satisfy this cut, a point
  must belong to one of the intervals given by the pairs. Each pair gives 
  lower and upper interval bounds.
*/
typedef std::vector<SprInterval> SprCut;


/*
  SprBox represents a box in a multidimensional space. The map index is
  the input dimension, and the map element is the pair of lower and upper
  bounds in this dimension.
*/
typedef std::map<unsigned,SprInterval> SprBox;


/*
  SprGrid is a set of multidimensional boxes. The conventions are given 
  by SprCut and SprBox.
*/
typedef std::map<unsigned,SprCut> SprGrid;

/*
  Structure for storing classifiers and their cuts.
*/
typedef std::pair<SprAbsClassifier*,SprCut> SprCCPair;

#endif

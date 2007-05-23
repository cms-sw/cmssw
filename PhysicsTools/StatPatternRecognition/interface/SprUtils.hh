//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprUtils.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprUtils :
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
 
#ifndef _SprUtils_HH
#define _SprUtils_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <utility>
#include <limits>
#include <algorithm>


struct SprUtils
{
  static SprCut lowerBound(double b) {
    return SprCut(1,std::pair<double,
		  double>(b,std::numeric_limits<double>::max()));
  }

  static SprCut upperBound(double b) {
    return SprCut(1,std::pair<double,
		  double>(-std::numeric_limits<double>::max(),b));
  }

  static SprCut inftyRange() {
    return SprCut(1,std::pair<double,
		  double>(-std::numeric_limits<double>::max(),
			   std::numeric_limits<double>::max()));
  }

  static double eps() {
    return std::numeric_limits<double>::epsilon();
  }

  static double min() {
    return -std::numeric_limits<double>::max();
  }

  static double max() {
    return std::numeric_limits<double>::max();
  }
};


/*
  Invokes std::stable_sort after placing all minimal elements at the
  beginning of the sequence. The goal of this method is to speed up sorting
  if many missing values are present in the data. It is assumed that
  missing values are modeled as negative numbers outside the range of
  physical values. If no missing values are present, the algorithm is
  a bit slower than std::stable_sort, of course, but not by much - the 
  additional cost introduced by this method goes as O(N) while 
  std::stable_sort is expected to perform between O(N*log(N)) and
  O(N*log(N)*log(N)).
*/
template <class Ran, class Cmp> void SprSort(Ran first, 
					     Ran last, 
					     Cmp cmp)
{
  // find minimal element
  Ran smallest = std::min_element(first,last,cmp);

  // place all minimal elements at the beginning
  Ran start = first;
  for( Ran iter=first;iter!=last;iter++ ) {
    if( !cmp(*iter,*smallest) && !cmp(*smallest,*iter) )
      std::iter_swap(start++,iter);
  }

  // sort the rest
  std::stable_sort(start,last,cmp);
}

#endif

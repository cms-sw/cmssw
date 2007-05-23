// File and Version Information:
//      $Id: SprAbsTrainedCombiner.hh,v 1.3 2006/11/13 19:09:38 narsky Exp $
//
// Description:
//      Class SprAbsTrainedCombiner :
//          Interface for trained combiners of classifiers.
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
 
#ifndef _SprAbsTrainedCombiner_HH
#define _SprAbsTrainedCombiner_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"

#include <cassert>
#include <vector>


class SprAbsTrainedCombiner : public SprAbsTrainedClassifier
{
public:
  virtual ~SprAbsTrainedCombiner() {}

  SprAbsTrainedCombiner(const std::vector<const SprAbsTrainedClassifier*>& c) 
    : 
    SprAbsTrainedClassifier(),
    classifiers_(c)
  {
    assert( !classifiers_.empty() );
  }

  SprAbsTrainedCombiner(const SprAbsTrainedCombiner& other)
    : 
    SprAbsTrainedClassifier(other),
    classifiers_(other.classifiers_)
  {}

  /*
    Combiner response for a data point. 
  */
  bool features(const std::vector<double>& v,
		std::vector<double>& features) const;

protected:
  std::vector<const SprAbsTrainedClassifier*> classifiers_;
};

#endif

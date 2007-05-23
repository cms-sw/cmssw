//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprFomCalculator.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprFomCalculator :
//         Computes FOM for specified data and classifier.
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
 
#ifndef _SprFomCalculator_HH
#define _SprFomCalculator_HH

class SprAbsFilter;
class SprAbsTwoClassCriterion;
class SprAverageLoss;
class SprAbsTrainedClassifier;
class SprClass;


class SprFomCalculator
{
public:
  virtual ~SprFomCalculator() {}

  SprFomCalculator() {}

  static double fom(const SprAbsFilter* data, const SprAbsTrainedClassifier* t,
		    const SprAbsTwoClassCriterion* crit, 
		    SprAverageLoss* loss, 
		    const SprClass& cls0, const SprClass& cls1);
};

#endif


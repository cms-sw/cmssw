//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprCrossValidator.hh,v 1.3 2006/11/13 19:09:39 narsky Exp $
//
// Description:
//      Class SprCrossValidator :
//         Cross-validates data by dividing the data into a specified
//         number of equal-sized pieces, training classifiers on all
//         pieces but one and computing FOM for the leftover piece.
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
 
#ifndef _SprCrossValidator_HH
#define _SprCrossValidator_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <vector>
#include <cassert>

class SprAbsFilter;
class SprEmptyFilter;
class SprAbsTwoClassCriterion;
class SprAverageLoss;
class SprAbsClassifier;
class SprClass;


class SprCrossValidator
{
public:
  virtual ~SprCrossValidator();

  SprCrossValidator(const SprAbsFilter* data, unsigned nPieces)
    : 
    data_(data),
    samples_(nPieces)
  {
    bool status = this->divide(nPieces);
    assert( status );
  }

  /*
    Returns a vector of cross-validated FOMs for the vector
    of supplied classifiers. Note that, due to some technicalities, this method
    works under assumption that the content of the supplied data has not
    changed since the time SprCrossValidator was constructed.
  */
  bool validate(const SprAbsTwoClassCriterion* crit,
		SprAverageLoss* loss,
		const std::vector<SprAbsClassifier*>& classifiers,
		const SprClass& cls0, const SprClass& cls1,
		const SprCut& cut,
		std::vector<double>& crossFom,
		int verbose=0) const;

private:
  // methods
  bool divide(unsigned nPieces);

  // data
  const SprAbsFilter* data_;
  std::vector<SprEmptyFilter*> samples_;
};

#endif


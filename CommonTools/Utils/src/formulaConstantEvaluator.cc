// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     reco::formula::ConstantEvaluator
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Sep 2015 18:06:29 GMT
//

// system include files

// user include files
#include "formulaConstantEvaluator.h"


namespace reco {
  namespace formula {
    double ConstantEvaluator::evaluate(double const* /*iVariables*/, double const* /*iParameters*/) const {
      return m_value;
    }
  }
}

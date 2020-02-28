// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     reco::formula::VariableEvaluator
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Sep 2015 18:06:29 GMT
//

// system include files

// user include files
#include "formulaVariableEvaluator.h"

namespace reco {
  namespace formula {
    double VariableEvaluator::evaluate(double const* iVariables, double const* /*iParameters*/) const {
      return iVariables[m_index];
    }

    std::vector<std::string> VariableEvaluator::abstractSyntaxTree() const {
      return std::vector<std::string>{1, std::string("var[") + std::to_string(m_index) + "]"};
    }
  }  // namespace formula
}  // namespace reco

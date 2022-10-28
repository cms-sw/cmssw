#ifndef CommonTools_Utils_formulaUnaryMinusEvaluator_h
#define CommonTools_Utils_formulaUnaryMinusEvaluator_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     formulaUnaryMinusEvaluator
//
/**\class reco::formula::UnaryMinusEvaluator formulaUnaryMinusEvaluator.h "formulaUnaryMinusEvaluator.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Sep 2015 17:41:33 GMT
//

// system include files
#include <memory>
#include <functional>

// user include files
#include "formulaEvaluatorBase.h"

// forward declarations

namespace reco {
  namespace formula {
    class UnaryMinusEvaluator : public EvaluatorBase {
    public:
      explicit UnaryMinusEvaluator(std::shared_ptr<EvaluatorBase> iArg)
          : EvaluatorBase(Precedence::kUnaryMinusOperator), m_arg(std::move(iArg)) {}

      // ---------- const member functions ---------------------
      double evaluate(double const* iVariables, double const* iParameters) const final {
        return -1. * m_arg->evaluate(iVariables, iParameters);
      }
      std::vector<std::string> abstractSyntaxTree() const final {
        auto ret = shiftAST(m_arg->abstractSyntaxTree());
        ret.emplace(ret.begin(), "unary minus");
        return ret;
      }

      UnaryMinusEvaluator(const UnaryMinusEvaluator&) = delete;

      const UnaryMinusEvaluator& operator=(const UnaryMinusEvaluator&) = delete;

    private:
      // ---------- member data --------------------------------
      std::shared_ptr<EvaluatorBase> m_arg;
    };
  }  // namespace formula
}  // namespace reco

#endif

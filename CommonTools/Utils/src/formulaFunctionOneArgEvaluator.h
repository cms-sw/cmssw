#ifndef CommonTools_Utils_formulaFunctionOneArgEvaluator_h
#define CommonTools_Utils_formulaFunctionOneArgEvaluator_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     formulaFunctionOneArgEvaluator
//
/**\class reco::formula::FunctionOneArgEvaluator formulaFunctionOneArgEvaluator.h "formulaFunctionOneArgEvaluator.h"

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
    class FunctionOneArgEvaluator : public EvaluatorBase {
    public:
      template <typename T>
      explicit FunctionOneArgEvaluator(std::shared_ptr<EvaluatorBase> iArg, T iFunc)
          : m_arg(std::move(iArg)), m_function(iFunc) {}

      // ---------- const member functions ---------------------
      double evaluate(double const* iVariables, double const* iParameters) const final {
        return m_function(m_arg->evaluate(iVariables, iParameters));
      }
      std::vector<std::string> abstractSyntaxTree() const final {
        auto ret = shiftAST(m_arg->abstractSyntaxTree());
        ret.emplace(ret.begin(), "func 1 arg");
        return ret;
      }

    private:
      FunctionOneArgEvaluator(const FunctionOneArgEvaluator&) = delete;

      const FunctionOneArgEvaluator& operator=(const FunctionOneArgEvaluator&) = delete;

      // ---------- member data --------------------------------
      std::shared_ptr<EvaluatorBase> m_arg;
      std::function<double(double)> m_function;
    };
  }  // namespace formula
}  // namespace reco

#endif

#ifndef CommonTools_Utils_formulaFunctionTwoArgsEvaluator_h
#define CommonTools_Utils_formulaFunctionTwoArgsEvaluator_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     formulaFunctionTwoArgsEvaluator
// 
/**\class reco::formula::FunctionTwoArgsEvaluator formulaFunctionTwoArgsEvaluator.h "formulaFunctionTwoArgsEvaluator.h"

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
    class FunctionTwoArgsEvaluator : public EvaluatorBase
    {
      
    public:
      template<typename T>
    FunctionTwoArgsEvaluator(std::shared_ptr<EvaluatorBase> iArg1,
                             std::shared_ptr<EvaluatorBase> iArg2,
                             T iFunc):
      m_arg1(std::move(iArg1)),
        m_arg2(std::move(iArg2)),
        m_function(iFunc)
        {}

      // ---------- const member functions ---------------------
      double evaluate(double const* iVariables, double const* iParameters) const final {
        return m_function( m_arg1->evaluate(iVariables,iParameters),
                           m_arg2->evaluate(iVariables,iParameters) );
      }
      std::vector<std::string> abstractSyntaxTree() const final {
        auto ret = shiftAST(m_arg1->abstractSyntaxTree());
        for(auto& v: shiftAST(m_arg2->abstractSyntaxTree()) ) {
          ret.emplace_back(std::move(v));
        }
        ret.emplace(ret.begin(), "func 2 args");
        return ret;
      }

    private:
      FunctionTwoArgsEvaluator(const FunctionTwoArgsEvaluator&) = delete;
      
      const FunctionTwoArgsEvaluator& operator=(const FunctionTwoArgsEvaluator&) = delete;
      
      // ---------- member data --------------------------------
      std::shared_ptr<EvaluatorBase> m_arg1;
      std::shared_ptr<EvaluatorBase> m_arg2;
      std::function<double(double,double)> m_function;
    };
  }
}


#endif

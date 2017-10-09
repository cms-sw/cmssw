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
    class FunctionOneArgEvaluator : public EvaluatorBase
    {
      
    public:
      template<typename T>
      explicit FunctionOneArgEvaluator(std::shared_ptr<EvaluatorBase> iArg, T iFunc):
        m_arg(std::move(iArg)),
        m_function(iFunc) {}
       
      // ---------- const member functions ---------------------
      virtual double evaluate(double const* iVariables, double const* iParameters) const override final {
        return m_function( m_arg->evaluate(iVariables,iParameters) );
      }

    private:
      FunctionOneArgEvaluator(const FunctionOneArgEvaluator&) = delete;
      
      const FunctionOneArgEvaluator& operator=(const FunctionOneArgEvaluator&) = delete;
      
      // ---------- member data --------------------------------
      std::shared_ptr<EvaluatorBase> m_arg;
      std::function<double(double)> m_function;
    };
  }
}


#endif

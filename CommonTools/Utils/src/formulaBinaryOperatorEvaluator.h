#ifndef CommonTools_Utils_formulaBinaryOperatorEvaluator_h
#define CommonTools_Utils_formulaBinaryOperatorEvaluator_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     formulaBinaryOperatorEvaluator
// 
/**\class reco::formula::BinaryOperatorEvaluator formulaBinaryOperatorEvaluator.h "formulaBinaryOperatorEvaluator.h"

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

// user include files
#include "formulaEvaluatorBase.h"

// forward declarations

namespace reco {
  namespace formula {
    class BinaryOperatorEvaluatorBase : public EvaluatorBase {
    public:
    BinaryOperatorEvaluatorBase( Precidence iPrec) :
      EvaluatorBase(iPrec) {}
      virtual void swapLeftEvaluator(std::unique_ptr<EvaluatorBase>& iNew) = 0;
    };
    template<typename Op>
      class BinaryOperatorEvaluator : public BinaryOperatorEvaluatorBase
    {
      
    public:
      BinaryOperatorEvaluator(std::unique_ptr<EvaluatorBase> iLHS, 
                              std::unique_ptr<EvaluatorBase> iRHS,
                              Precidence iPrec):
      BinaryOperatorEvaluatorBase(iPrec),
        m_lhs(std::move(iLHS)),
        m_rhs(std::move(iRHS)) {
        }
       
      // ---------- const member functions ---------------------
      virtual double evaluate(double const* iVariables, double const* iParameters) const override final {
        return m_operator(m_lhs->evaluate(iVariables,iParameters),m_rhs->evaluate(iVariables,iParameters));
      }

      void swapLeftEvaluator(std::unique_ptr<EvaluatorBase>& iNew ) override final {
        m_lhs.swap(iNew);
      }

    private:
      BinaryOperatorEvaluator(const BinaryOperatorEvaluator&) = delete;
      
      const BinaryOperatorEvaluator& operator=(const BinaryOperatorEvaluator&) = delete;
      
      // ---------- member data --------------------------------
      std::unique_ptr<EvaluatorBase> m_lhs;
      std::unique_ptr<EvaluatorBase> m_rhs;
      Op m_operator;

    };
  }
}


#endif

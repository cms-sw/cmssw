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
      BinaryOperatorEvaluatorBase(std::shared_ptr<EvaluatorBase> iLHS,
                                  std::shared_ptr<EvaluatorBase> iRHS,
                                  Precedence iPrec)
          : EvaluatorBase(iPrec), m_lhs(iLHS), m_rhs(iRHS) {}

      BinaryOperatorEvaluatorBase(Precedence iPrec) : EvaluatorBase(iPrec) {}

      void swapLeftEvaluator(std::shared_ptr<EvaluatorBase>& iNew) { m_lhs.swap(iNew); }

      void setLeftEvaluator(std::shared_ptr<EvaluatorBase> iOther) { m_lhs = std::move(iOther); }
      void setRightEvaluator(std::shared_ptr<EvaluatorBase> iOther) { m_rhs = std::move(iOther); }

      EvaluatorBase const* lhs() const { return m_lhs.get(); }
      EvaluatorBase const* rhs() const { return m_rhs.get(); }

    private:
      std::shared_ptr<EvaluatorBase> m_lhs;
      std::shared_ptr<EvaluatorBase> m_rhs;
    };

    template <typename Op>
    class BinaryOperatorEvaluator : public BinaryOperatorEvaluatorBase {
    public:
      BinaryOperatorEvaluator(std::shared_ptr<EvaluatorBase> iLHS,
                              std::shared_ptr<EvaluatorBase> iRHS,
                              Precedence iPrec)
          : BinaryOperatorEvaluatorBase(std::move(iLHS), std::move(iRHS), iPrec) {}

      BinaryOperatorEvaluator(Precedence iPrec) : BinaryOperatorEvaluatorBase(iPrec) {}

      // ---------- const member functions ---------------------
      double evaluate(double const* iVariables, double const* iParameters) const final {
        return m_operator(lhs()->evaluate(iVariables, iParameters), rhs()->evaluate(iVariables, iParameters));
      }

      std::vector<std::string> abstractSyntaxTree() const final {
        std::vector<std::string> ret;
        if (lhs()) {
          ret = shiftAST(lhs()->abstractSyntaxTree());
        } else {
          ret.emplace_back(".nullptr");
        }
        if (rhs()) {
          auto child = shiftAST(rhs()->abstractSyntaxTree());
          for (auto& v : child) {
            ret.emplace_back(std::move(v));
          }
        } else {
          ret.emplace_back(".nullptr");
        }
        ret.emplace(ret.begin(), std::string("op ") + std::to_string(precedence()));
        return ret;
      }

    private:
      BinaryOperatorEvaluator(const BinaryOperatorEvaluator&) = delete;

      const BinaryOperatorEvaluator& operator=(const BinaryOperatorEvaluator&) = delete;

      // ---------- member data --------------------------------
      Op m_operator;
    };
  }  // namespace formula
}  // namespace reco

#endif

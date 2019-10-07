// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     FormulaEvaluator
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 24 Sep 2015 19:07:58 GMT
//

// system include files
#include <cassert>
#include <functional>
#include <cstdlib>
#include <cmath>
#include "TMath.h"

//#define DEBUG_AST
#if defined(DEBUG_AST)
#include <iostream>
#endif
// user include files
#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include "formulaEvaluatorBase.h"
#include "formulaUnaryMinusEvaluator.h"
#include "formulaBinaryOperatorEvaluator.h"
#include "formulaConstantEvaluator.h"
#include "formulaVariableEvaluator.h"
#include "formulaParameterEvaluator.h"
#include "formulaFunctionOneArgEvaluator.h"
#include "formulaFunctionTwoArgsEvaluator.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

namespace {

#if defined(DEBUG_AST)
  void printAST(formula::EvaluatorBase* e) {
    std::cout << "printAST" << std::endl;
    for (auto const& n : e->abstractSyntaxTree()) {
      std::cout << n << std::endl;
    }
  }
#define DEBUG_STATE(_v_) std::cout << _v_ << std::endl
#else
  inline void printAST(void*) {}
#define DEBUG_STATE(_v_)
#endif
  //Formula Parser Code
  struct EvaluatorInfo {
    std::shared_ptr<reco::formula::EvaluatorBase> evaluator;
    std::shared_ptr<reco::formula::EvaluatorBase> top;
    int nextParseIndex = 0;
    unsigned int maxNumVariables = 0;
    unsigned int maxNumParameters = 0;
  };

  class ExpressionElementFinderBase {
  public:
    virtual bool checkStart(char) const = 0;

    virtual EvaluatorInfo createEvaluator(std::string::const_iterator, std::string::const_iterator) const = 0;

    virtual ~ExpressionElementFinderBase() = default;
  };

  std::string::const_iterator findMatchingParenthesis(std::string::const_iterator iBegin,
                                                      std::string::const_iterator iEnd) {
    if (iBegin == iEnd) {
      return iBegin;
    }
    if (*iBegin != '(') {
      return iBegin;
    }
    int level = 1;
    size_t index = 0;
    for (auto it = iBegin + 1; it != iEnd; ++it) {
      ++index;
      if (*it == '(') {
        ++level;
      } else if (*it == ')') {
        --level;
        if (level == 0) {
          break;
        }
      }
    }
    return iBegin + index;
  }

  class ConstantFinder : public ExpressionElementFinderBase {
    bool checkStart(char iSymbol) const final {
      if (iSymbol == '-' or iSymbol == '.' or std::isdigit(iSymbol)) {
        return true;
      }
      return false;
    }

    EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const final {
      EvaluatorInfo info;
      try {
        size_t endIndex = 0;
        std::string s(iBegin, iEnd);
        double value = stod(s, &endIndex);

        info.nextParseIndex = endIndex;
        info.evaluator = std::make_shared<reco::formula::ConstantEvaluator>(value);
        info.top = info.evaluator;
      } catch (std::invalid_argument const&) {
      }

      return info;
    }
  };

  class ParameterFinder : public ExpressionElementFinderBase {
    bool checkStart(char iSymbol) const final {
      if (iSymbol == '[') {
        return true;
      }
      return false;
    }

    EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const final {
      EvaluatorInfo info;
      if (iEnd == iBegin) {
        return info;
      }
      if (*iBegin != '[') {
        return info;
      }
      info.nextParseIndex = 1;
      try {
        size_t endIndex = 0;
        std::string s(iBegin + 1, iEnd);
        unsigned long value = stoul(s, &endIndex);

        if (iBegin + endIndex + 1 == iEnd or *(iBegin + 1 + endIndex) != ']') {
          return info;
        }

        info.nextParseIndex = endIndex + 2;
        info.maxNumParameters = value + 1;
        info.evaluator = std::make_shared<reco::formula::ParameterEvaluator>(value);
        info.top = info.evaluator;
      } catch (std::invalid_argument const&) {
      }

      return info;
    }
  };

  class VariableFinder : public ExpressionElementFinderBase {
    bool checkStart(char iSymbol) const final {
      if (iSymbol == 'x' or iSymbol == 'y' or iSymbol == 'z' or iSymbol == 't') {
        return true;
      }
      return false;
    }

    EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const final {
      EvaluatorInfo info;
      if (iBegin == iEnd) {
        return info;
      }
      unsigned int index = 4;
      switch (*iBegin) {
        case 'x': {
          index = 0;
          break;
        }
        case 'y': {
          index = 1;
          break;
        }
        case 'z': {
          index = 2;
          break;
        }
        case 't': {
          index = 3;
          break;
        }
      }
      if (index == 4) {
        return info;
      }
      info.nextParseIndex = 1;
      info.maxNumVariables = index + 1;
      info.evaluator = std::make_shared<reco::formula::VariableEvaluator>(index);
      info.top = info.evaluator;
      return info;
    }
  };

  class ExpressionFinder;

  class FunctionFinder : public ExpressionElementFinderBase {
  public:
    FunctionFinder(ExpressionFinder const* iEF) : m_expressionFinder(iEF){};

    bool checkStart(char iSymbol) const final { return std::isalpha(iSymbol); }

    EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const final;

  private:
    ExpressionFinder const* m_expressionFinder;
  };

  EvaluatorInfo createBinaryOperatorEvaluator(ExpressionFinder const&,
                                              std::string::const_iterator iBegin,
                                              std::string::const_iterator iEnd);

  class ExpressionFinder {
  public:
    ExpressionFinder() {
      m_elements.reserve(4);
      m_elements.emplace_back(new FunctionFinder{this});
      m_elements.emplace_back(new ConstantFinder{});
      m_elements.emplace_back(new ParameterFinder{});
      m_elements.emplace_back(new VariableFinder{});
    }

    bool checkStart(char iChar) const {
      if ('(' == iChar or '-' == iChar or '+' == iChar) {
        return true;
      }
      for (auto const& e : m_elements) {
        if (e->checkStart(iChar)) {
          return true;
        }
      }
      return false;
    }

    EvaluatorInfo createEvaluator(std::string::const_iterator iBegin,
                                  std::string::const_iterator iEnd,
                                  std::shared_ptr<reco::formula::BinaryOperatorEvaluatorBase> iPreviousBinary) const {
      EvaluatorInfo leftEvaluatorInfo;

      if (iBegin == iEnd) {
        return leftEvaluatorInfo;
      }
      //Start with '+'
      if (*iBegin == '+' and iEnd - iBegin > 1 and not std::isdigit(*(iBegin + 1))) {
        leftEvaluatorInfo = createEvaluator(iBegin + 1, iEnd, iPreviousBinary);

        //have to account for the '+' we skipped over
        leftEvaluatorInfo.nextParseIndex += 1;
        if (nullptr == leftEvaluatorInfo.evaluator.get()) {
          return leftEvaluatorInfo;
        }
      }
      //Start with '-'
      else if (*iBegin == '-' and iEnd - iBegin > 1 and not std::isdigit(*(iBegin + 1))) {
        leftEvaluatorInfo = createEvaluator(iBegin + 1, iEnd, iPreviousBinary);

        //have to account for the '+' we skipped over
        leftEvaluatorInfo.nextParseIndex += 1;
        if (nullptr == leftEvaluatorInfo.evaluator.get()) {
          return leftEvaluatorInfo;
        }
        leftEvaluatorInfo.evaluator =
            std::make_shared<reco::formula::UnaryMinusEvaluator>(std::move(leftEvaluatorInfo.top));
        leftEvaluatorInfo.top = leftEvaluatorInfo.evaluator;
      }
      //Start with '('
      else if (*iBegin == '(') {
        auto endParenthesis = findMatchingParenthesis(iBegin, iEnd);
        if (iBegin == endParenthesis) {
          return leftEvaluatorInfo;
        }
        leftEvaluatorInfo =
            createEvaluator(iBegin + 1, endParenthesis, std::shared_ptr<reco::formula::BinaryOperatorEvaluatorBase>());
        ++leftEvaluatorInfo.nextParseIndex;
        if (leftEvaluatorInfo.evaluator.get() == nullptr) {
          return leftEvaluatorInfo;
        }
        //need to account for closing parenthesis
        ++leftEvaluatorInfo.nextParseIndex;
        leftEvaluatorInfo.top->setPrecedenceToParenthesis();
        DEBUG_STATE("close parenthesis");
        printAST(leftEvaluatorInfo.top.get());
        leftEvaluatorInfo.evaluator = leftEvaluatorInfo.top;
      } else {
        //Does not start with a '('
        int maxParseDistance = 0;
        for (auto const& e : m_elements) {
          if (e->checkStart(*iBegin)) {
            leftEvaluatorInfo = e->createEvaluator(iBegin, iEnd);
            if (leftEvaluatorInfo.evaluator != nullptr) {
              break;
            }
            if (leftEvaluatorInfo.nextParseIndex > maxParseDistance) {
              maxParseDistance = leftEvaluatorInfo.nextParseIndex;
            }
          }
        }
        if (leftEvaluatorInfo.evaluator.get() == nullptr) {
          //failed to parse
          leftEvaluatorInfo.nextParseIndex = maxParseDistance;
          return leftEvaluatorInfo;
        }
      }
      //did we evaluate the full expression?
      if (leftEvaluatorInfo.nextParseIndex == iEnd - iBegin) {
        if (iPreviousBinary) {
          iPreviousBinary->setRightEvaluator(leftEvaluatorInfo.top);
          leftEvaluatorInfo.top = iPreviousBinary;
        }
        DEBUG_STATE("full expression");
        printAST(leftEvaluatorInfo.evaluator.get());
        return leftEvaluatorInfo;
      }

      //see if this is a binary expression
      auto fullExpression = createBinaryOperatorEvaluator(*this, iBegin + leftEvaluatorInfo.nextParseIndex, iEnd);
      fullExpression.nextParseIndex += leftEvaluatorInfo.nextParseIndex;
      fullExpression.maxNumVariables = std::max(leftEvaluatorInfo.maxNumVariables, fullExpression.maxNumVariables);
      fullExpression.maxNumParameters = std::max(leftEvaluatorInfo.maxNumParameters, fullExpression.maxNumParameters);
      if (iBegin + fullExpression.nextParseIndex != iEnd) {
        //did not parse the full expression
        fullExpression.evaluator.reset();
      }

      if (fullExpression.evaluator == nullptr) {
        //we had a parsing problem
        return fullExpression;
      }

      DEBUG_STATE("binary before precedence handling");
      printAST(fullExpression.evaluator.get());
      //Now to handle precedence
      auto topNode = fullExpression.top;
      auto binaryEval = dynamic_cast<reco::formula::BinaryOperatorEvaluatorBase*>(fullExpression.evaluator.get());
      if (iPreviousBinary) {
        if (iPreviousBinary->precedence() >= fullExpression.evaluator->precedence()) {
          DEBUG_STATE("prec >=");
          iPreviousBinary->setRightEvaluator(leftEvaluatorInfo.evaluator);
          binaryEval->setLeftEvaluator(iPreviousBinary);
        } else {
          binaryEval->setLeftEvaluator(leftEvaluatorInfo.evaluator);
          if (iPreviousBinary->precedence() < topNode->precedence()) {
            DEBUG_STATE(" switch topNode");
            topNode = iPreviousBinary;
            iPreviousBinary->setRightEvaluator(fullExpression.top);
          } else {
            DEBUG_STATE("swapping");
            //We need to take the lhs of a  binary expression directly or indirectly connected
            // to the present node and swap it with the rhs of the 'previous' binary expression
            // becuase we need the present expression to be evaluated earlier than the 'previous'.
            std::shared_ptr<reco::formula::EvaluatorBase> toSwap = iPreviousBinary;
            auto parentBinary = dynamic_cast<reco::formula::BinaryOperatorEvaluatorBase*>(topNode.get());
            do {
              if (parentBinary->lhs() == binaryEval or
                  parentBinary->lhs()->precedence() > iPreviousBinary->precedence()) {
                parentBinary->swapLeftEvaluator(toSwap);
                iPreviousBinary->setRightEvaluator(toSwap);
              } else {
                //try the next one in the chain
                parentBinary = const_cast<reco::formula::BinaryOperatorEvaluatorBase*>(
                    dynamic_cast<const reco::formula::BinaryOperatorEvaluatorBase*>(parentBinary->lhs()));
                assert(parentBinary != nullptr);
              }
            } while (iPreviousBinary->rhs() == nullptr);
          }
        }
      } else {
        binaryEval->setLeftEvaluator(leftEvaluatorInfo.top);
      }
      DEBUG_STATE("finished binary");
      printAST(binaryEval);
      DEBUG_STATE("present top");
      printAST(topNode.get());
      fullExpression.top = topNode;
      return fullExpression;
    }

  private:
    std::vector<std::unique_ptr<ExpressionElementFinderBase>> m_elements;
  };

  template <typename Op>
  EvaluatorInfo createBinaryOperatorEvaluatorT(int iSymbolLength,
                                               reco::formula::EvaluatorBase::Precedence iPrec,
                                               ExpressionFinder const& iEF,
                                               std::string::const_iterator iBegin,
                                               std::string::const_iterator iEnd) {
    auto op = std::make_shared<reco::formula::BinaryOperatorEvaluator<Op>>(iPrec);
    EvaluatorInfo evalInfo = iEF.createEvaluator(iBegin + iSymbolLength, iEnd, op);
    evalInfo.nextParseIndex += iSymbolLength;

    if (evalInfo.evaluator.get() == nullptr) {
      return evalInfo;
    }

    evalInfo.evaluator = op;
    return evalInfo;
  }

  struct power {
    double operator()(double iLHS, double iRHS) const { return std::pow(iLHS, iRHS); }
  };

  EvaluatorInfo createBinaryOperatorEvaluator(ExpressionFinder const& iEF,
                                              std::string::const_iterator iBegin,
                                              std::string::const_iterator iEnd) {
    EvaluatorInfo evalInfo;
    if (iBegin == iEnd) {
      return evalInfo;
    }

    if (*iBegin == '+') {
      return createBinaryOperatorEvaluatorT<std::plus<double>>(
          1, reco::formula::EvaluatorBase::Precedence::kPlusMinus, iEF, iBegin, iEnd);
    }

    else if (*iBegin == '-') {
      return createBinaryOperatorEvaluatorT<std::minus<double>>(
          1, reco::formula::EvaluatorBase::Precedence::kPlusMinus, iEF, iBegin, iEnd);
    } else if (*iBegin == '*') {
      return createBinaryOperatorEvaluatorT<std::multiplies<double>>(
          1, reco::formula::EvaluatorBase::Precedence::kMultDiv, iEF, iBegin, iEnd);
    } else if (*iBegin == '/') {
      return createBinaryOperatorEvaluatorT<std::divides<double>>(
          1, reco::formula::EvaluatorBase::Precedence::kMultDiv, iEF, iBegin, iEnd);
    }

    else if (*iBegin == '^') {
      return createBinaryOperatorEvaluatorT<power>(
          1, reco::formula::EvaluatorBase::Precedence::kPower, iEF, iBegin, iEnd);
    } else if (*iBegin == '<' and iBegin + 1 != iEnd and *(iBegin + 1) == '=') {
      return createBinaryOperatorEvaluatorT<std::less_equal<double>>(
          2, reco::formula::EvaluatorBase::Precedence::kComparison, iEF, iBegin, iEnd);

    } else if (*iBegin == '>' and iBegin + 1 != iEnd and *(iBegin + 1) == '=') {
      return createBinaryOperatorEvaluatorT<std::greater_equal<double>>(
          2, reco::formula::EvaluatorBase::Precedence::kComparison, iEF, iBegin, iEnd);

    } else if (*iBegin == '<') {
      return createBinaryOperatorEvaluatorT<std::less<double>>(
          1, reco::formula::EvaluatorBase::Precedence::kComparison, iEF, iBegin, iEnd);

    } else if (*iBegin == '>') {
      return createBinaryOperatorEvaluatorT<std::greater<double>>(
          1, reco::formula::EvaluatorBase::Precedence::kComparison, iEF, iBegin, iEnd);

    } else if (*iBegin == '=' and iBegin + 1 != iEnd and *(iBegin + 1) == '=') {
      return createBinaryOperatorEvaluatorT<std::equal_to<double>>(
          2, reco::formula::EvaluatorBase::Precedence::kIdentity, iEF, iBegin, iEnd);

    } else if (*iBegin == '!' and iBegin + 1 != iEnd and *(iBegin + 1) == '=') {
      return createBinaryOperatorEvaluatorT<std::not_equal_to<double>>(
          2, reco::formula::EvaluatorBase::Precedence::kIdentity, iEF, iBegin, iEnd);
    }
    return evalInfo;
  }

  template <typename Op>
  EvaluatorInfo checkForSingleArgFunction(std::string::const_iterator iBegin,
                                          std::string::const_iterator iEnd,
                                          ExpressionFinder const* iExpressionFinder,
                                          const std::string& iName,
                                          Op op) {
    EvaluatorInfo info;
    if (iName.size() + 2 > static_cast<unsigned int>(iEnd - iBegin)) {
      return info;
    }
    auto pos = iName.find(&(*iBegin), 0, iName.size());

    if (std::string::npos == pos or *(iBegin + iName.size()) != '(') {
      return info;
    }

    info.nextParseIndex = iName.size() + 1;

    auto itEndParen = findMatchingParenthesis(iBegin + iName.size(), iEnd);
    if (iBegin + iName.size() == itEndParen) {
      return info;
    }

    auto argEvaluatorInfo = iExpressionFinder->createEvaluator(
        iBegin + iName.size() + 1, itEndParen, std::shared_ptr<reco::formula::BinaryOperatorEvaluatorBase>());
    info.nextParseIndex += argEvaluatorInfo.nextParseIndex;
    if (argEvaluatorInfo.evaluator.get() == nullptr or info.nextParseIndex + 1 != 1 + itEndParen - iBegin) {
      return info;
    }
    //account for closing parenthesis
    ++info.nextParseIndex;

    info.evaluator = std::make_shared<reco::formula::FunctionOneArgEvaluator>(std::move(argEvaluatorInfo.top), op);
    info.top = info.evaluator;
    return info;
  }

  std::string::const_iterator findCommaNotInParenthesis(std::string::const_iterator iBegin,
                                                        std::string::const_iterator iEnd) {
    int level = 0;
    std::string::const_iterator it = iBegin;
    for (; it != iEnd; ++it) {
      if (*it == '(') {
        ++level;
      } else if (*it == ')') {
        --level;
      } else if (*it == ',' and level == 0) {
        return it;
      }
    }

    return it;
  }

  template <typename Op>
  EvaluatorInfo checkForTwoArgsFunction(std::string::const_iterator iBegin,
                                        std::string::const_iterator iEnd,
                                        ExpressionFinder const* iExpressionFinder,
                                        const std::string& iName,
                                        Op op) {
    EvaluatorInfo info;
    if (iName.size() + 2 > static_cast<unsigned int>(iEnd - iBegin)) {
      return info;
    }
    auto pos = iName.find(&(*iBegin), 0, iName.size());

    if (std::string::npos == pos or *(iBegin + iName.size()) != '(') {
      return info;
    }

    info.nextParseIndex = iName.size() + 1;

    auto itEndParen = findMatchingParenthesis(iBegin + iName.size(), iEnd);
    if (iBegin + iName.size() == itEndParen) {
      return info;
    }

    auto itComma = findCommaNotInParenthesis(iBegin + iName.size() + 1, itEndParen);

    auto arg1EvaluatorInfo = iExpressionFinder->createEvaluator(
        iBegin + iName.size() + 1, itComma, std::shared_ptr<reco::formula::BinaryOperatorEvaluatorBase>());
    info.nextParseIndex += arg1EvaluatorInfo.nextParseIndex;
    if (arg1EvaluatorInfo.evaluator.get() == nullptr or info.nextParseIndex != itComma - iBegin) {
      return info;
    }
    //account for commas
    ++info.nextParseIndex;

    auto arg2EvaluatorInfo = iExpressionFinder->createEvaluator(
        itComma + 1, itEndParen, std::shared_ptr<reco::formula::BinaryOperatorEvaluatorBase>());
    info.nextParseIndex += arg2EvaluatorInfo.nextParseIndex;

    if (arg2EvaluatorInfo.evaluator.get() == nullptr or info.nextParseIndex + 1 != 1 + itEndParen - iBegin) {
      return info;
    }
    //account for closing parenthesis
    ++info.nextParseIndex;

    info.evaluator = std::make_shared<reco::formula::FunctionTwoArgsEvaluator>(
        std::move(arg1EvaluatorInfo.top), std::move(arg2EvaluatorInfo.top), op);
    info.top = info.evaluator;
    return info;
  }

  const std::string k_log("log");
  const std::string k_log10("log10");
  const std::string k_TMath__Log("TMath::Log");
  double const kLog10Inv = 1. / std::log(10.);
  const std::string k_exp("exp");
  const std::string k_pow("pow");
  const std::string k_TMath__Power("TMath::Power");
  const std::string k_max("max");
  const std::string k_min("min");
  const std::string k_TMath__Max("TMath::Max");
  const std::string k_TMath__Min("TMath::Min");
  const std::string k_TMath__Erf("TMath::Erf");
  const std::string k_erf("erf");
  const std::string k_TMath__Landau("TMath::Landau");
  const std::string k_sqrt("sqrt");
  const std::string k_TMath__Sqrt("TMath::Sqrt");
  const std::string k_abs("abs");
  const std::string k_TMath__Abs("TMath::Abs");
  const std::string k_cos("cos");
  const std::string k_TMath__Cos("TMath::Cos");
  const std::string k_sin("sin");
  const std::string k_TMath__Sin("TMath::Sin");
  const std::string k_tan("tan");
  const std::string k_TMath__Tan("TMath::Tan");
  const std::string k_acos("acos");
  const std::string k_TMath__ACos("TMath::ACos");
  const std::string k_asin("asin");
  const std::string k_TMath__ASin("TMath::ASin");
  const std::string k_atan("atan");
  const std::string k_TMath__ATan("TMath::ATan");
  const std::string k_atan2("atan2");
  const std::string k_TMath__ATan2("TMath::ATan2");
  const std::string k_cosh("cosh");
  const std::string k_TMath__CosH("TMath::CosH");
  const std::string k_sinh("sinh");
  const std::string k_TMath__SinH("TMath::SinH");
  const std::string k_tanh("tanh");
  const std::string k_TMath__TanH("TMath::TanH");
  const std::string k_acosh("acosh");
  const std::string k_TMath__ACosH("TMath::ACosH");
  const std::string k_asinh("asinh");
  const std::string k_TMath__ASinH("TMath::ASinH");
  const std::string k_atanh("atanh");
  const std::string k_TMath__ATanH("TMath::ATanH");

  EvaluatorInfo FunctionFinder::createEvaluator(std::string::const_iterator iBegin,
                                                std::string::const_iterator iEnd) const {
    EvaluatorInfo info;

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_erf, [](double iArg) -> double { return std::erf(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Erf, [](double iArg) -> double { return std::erf(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Landau, [](double iArg) -> double { return TMath::Landau(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_log, [](double iArg) -> double { return std::log(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Log, [](double iArg) -> double { return std::log(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_log10, [](double iArg) -> double { return std::log(iArg) * kLog10Inv; });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_exp, [](double iArg) -> double { return std::exp(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_sqrt, [](double iArg) -> double { return std::sqrt(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Sqrt, [](double iArg) -> double { return std::sqrt(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_abs, [](double iArg) -> double { return std::abs(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Abs, [](double iArg) -> double { return std::abs(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_cos, [](double iArg) -> double { return std::cos(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Cos, [](double iArg) -> double { return std::cos(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_sin, [](double iArg) -> double { return std::sin(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Sin, [](double iArg) -> double { return std::sin(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_tan, [](double iArg) -> double { return std::tan(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Tan, [](double iArg) -> double { return std::tan(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_acos, [](double iArg) -> double { return std::acos(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ACos, [](double iArg) -> double { return std::acos(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_asin, [](double iArg) -> double { return std::asin(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ASin, [](double iArg) -> double { return std::asin(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_atan, [](double iArg) -> double { return std::atan(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ATan, [](double iArg) -> double { return std::atan(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_cosh, [](double iArg) -> double { return std::cosh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__CosH, [](double iArg) -> double { return std::cosh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_sinh, [](double iArg) -> double { return std::sinh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__SinH, [](double iArg) -> double { return std::sinh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_tanh, [](double iArg) -> double { return std::tanh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__TanH, [](double iArg) -> double { return std::tanh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_acosh, [](double iArg) -> double { return std::acosh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ACosH, [](double iArg) -> double { return std::acosh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_asinh, [](double iArg) -> double { return std::asinh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ASinH, [](double iArg) -> double { return std::asinh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_atanh, [](double iArg) -> double { return std::atanh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ATanH, [](double iArg) -> double { return std::atanh(iArg); });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(iBegin, iEnd, m_expressionFinder, k_atan2, [](double iArg1, double iArg2) -> double {
      return std::atan2(iArg1, iArg2);
    });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__ATan2, [](double iArg1, double iArg2) -> double {
          return std::atan2(iArg1, iArg2);
        });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(iBegin, iEnd, m_expressionFinder, k_pow, [](double iArg1, double iArg2) -> double {
      return std::pow(iArg1, iArg2);
    });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Power, [](double iArg1, double iArg2) -> double {
          return std::pow(iArg1, iArg2);
        });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(iBegin, iEnd, m_expressionFinder, k_max, [](double iArg1, double iArg2) -> double {
      return std::max(iArg1, iArg2);
    });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(iBegin, iEnd, m_expressionFinder, k_min, [](double iArg1, double iArg2) -> double {
      return std::min(iArg1, iArg2);
    });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Max, [](double iArg1, double iArg2) -> double {
          return std::max(iArg1, iArg2);
        });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(
        iBegin, iEnd, m_expressionFinder, k_TMath__Min, [](double iArg1, double iArg2) -> double {
          return std::min(iArg1, iArg2);
        });
    if (info.evaluator.get() != nullptr) {
      return info;
    }

    return info;
  };

  ExpressionFinder const s_expressionFinder;

}  // namespace
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FormulaEvaluator::FormulaEvaluator(std::string const& iFormula) {
  //remove white space
  std::string formula;
  formula.reserve(iFormula.size());
  std::copy_if(iFormula.begin(), iFormula.end(), std::back_inserter(formula), [](const char iC) { return iC != ' '; });

  auto info = s_expressionFinder.createEvaluator(
      formula.begin(), formula.end(), std::shared_ptr<reco::formula::BinaryOperatorEvaluatorBase>());

  if (info.nextParseIndex != static_cast<int>(formula.size()) or info.top.get() == nullptr) {
    auto lastIndex = info.nextParseIndex;
    if (formula.size() != iFormula.size()) {
      lastIndex = 0;
      for (decltype(info.nextParseIndex) index = 0; index < info.nextParseIndex; ++index, ++lastIndex) {
        while (iFormula[lastIndex] != formula[index]) {
          assert(iFormula[lastIndex] == ' ');
          ++lastIndex;
        }
      }
    }
    throw cms::Exception("FormulaEvaluatorParseError")
        << "While parsing '" << iFormula << "' could not parse beyond '"
        << std::string(iFormula.begin(), iFormula.begin() + lastIndex) << "'";
  }

  DEBUG_STATE("DONE parsing");
  printAST(info.top.get());

  m_evaluator = std::move(info.top);
  m_nVariables = info.maxNumVariables;
  m_nParameters = info.maxNumParameters;
}

//
// const member functions
//
double FormulaEvaluator::evaluate(double const* iVariables, double const* iParameters) const {
  return m_evaluator->evaluate(iVariables, iParameters);
}

void FormulaEvaluator::throwWrongNumberOfVariables(size_t iSize) const {
  throw cms::Exception("WrongNumVariables")
      << "FormulaEvaluator expected at least " << m_nVariables << " but was passed only " << iSize;
}
void FormulaEvaluator::throwWrongNumberOfParameters(size_t iSize) const {
  throw cms::Exception("WrongNumParameters")
      << "FormulaEvaluator expected at least " << m_nParameters << " but was passed only " << iSize;
}

std::vector<std::string> FormulaEvaluator::abstractSyntaxTree() const { return m_evaluator->abstractSyntaxTree(); }

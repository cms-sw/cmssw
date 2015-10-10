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
  //Formula Parser Code
  struct EvaluatorInfo {
    std::unique_ptr<reco::formula::EvaluatorBase> evaluator;
    int nextParseIndex=0;
    unsigned int maxNumVariables=0;
    unsigned int maxNumParameters=0;
  };

  class ExpressionElementFinderBase {
  public:
    virtual bool checkStart(char) const = 0;

    virtual EvaluatorInfo createEvaluator(std::string::const_iterator, std::string::const_iterator) const = 0;
  };

  std::string::const_iterator findMatchingParenthesis(std::string::const_iterator iBegin, std::string::const_iterator iEnd) {
    if (iBegin == iEnd) {
      return iBegin;
    }
    if( *iBegin != '(') {
      return iBegin;
    }
    int level = 1;
    size_t index = 0;
    for( auto it = iBegin+1; it != iEnd; ++it) {
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
    virtual bool checkStart(char iSymbol) const override final {
      if( iSymbol == '-' or iSymbol == '.' or std::isdigit(iSymbol) ) {
        return true;
      }
      return false;
    }

    virtual EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const override final {
      EvaluatorInfo info;
      try {
        size_t endIndex=0;
        std::string s(iBegin,iEnd);
        double value = stod(s, &endIndex);

        info.nextParseIndex = endIndex;
        info.evaluator = std::unique_ptr<reco::formula::EvaluatorBase>( new reco::formula::ConstantEvaluator(value));
      } catch ( std::invalid_argument ) {}

      return info;

    }
  };


  class ParameterFinder : public ExpressionElementFinderBase {
    virtual bool checkStart(char iSymbol) const override final {
      if( iSymbol == '[') {
        return true;
      }
      return false;
    }

    virtual EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const override final {
      EvaluatorInfo info;
      if(iEnd == iBegin) {
        return info;
      }
      if(*iBegin != '[') {
        return info;
      }
      info.nextParseIndex = 1;
      try {
        size_t endIndex=0;
        std::string s(iBegin+1,iEnd);
        unsigned long value = stoul(s, &endIndex);
        
        if( iBegin+endIndex+1 == iEnd or *(iBegin+1+endIndex) != ']' ) {
          return info;
        }
        
        
        info.nextParseIndex = endIndex+2;
        info.maxNumParameters = value;
        info.evaluator = std::unique_ptr<reco::formula::EvaluatorBase>( new reco::formula::ParameterEvaluator(value));
      } catch ( std::invalid_argument ) {}
      
      return info;

    }
  };
  
  class VariableFinder : public ExpressionElementFinderBase {
    virtual bool checkStart(char iSymbol) const override final {
      if( iSymbol == 'x' or iSymbol == 'y' or iSymbol == 'z' or iSymbol == 't' ) {
        return true;
      }
      return false;
    }
    
    virtual EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const override final {
      EvaluatorInfo info;
      if(iBegin == iEnd) {
        return info;
      }
      unsigned int index = 4;
      switch (*iBegin) {
      case 'x':
        { index = 0; break;}
      case 'y':
        { index = 1; break;}
      case 'z':
        { index = 2; break;}
      case 't':
        {index = 3; break;}
      }
      if(index == 4) {
        return info;
      }
      info.nextParseIndex = 1;
      info.maxNumVariables = index+1;
      info.evaluator = std::unique_ptr<reco::formula::EvaluatorBase>(new reco::formula::VariableEvaluator(index) );
      return info;
    }
  };

  class ExpressionFinder;

  class FunctionFinder : public ExpressionElementFinderBase {
  public:
    FunctionFinder(ExpressionFinder const* iEF):
      m_expressionFinder(iEF) {};

    virtual bool checkStart(char iSymbol) const override final {
      return std::isalpha(iSymbol);
    }

    virtual EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const override final;

  private:
    ExpressionFinder const* m_expressionFinder;
  };


  EvaluatorInfo createBinaryOperatorEvaluator( ExpressionFinder const&,
                                               std::unique_ptr<reco::formula::EvaluatorBase> iLHS,
                                               std::string::const_iterator iBegin,
                                               std::string::const_iterator iEnd) ;

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
      if ( '(' == iChar or '-' == iChar or '+' ==iChar) {
        return true;
      }
      for( auto const& e : m_elements) {
        if (e->checkStart(iChar) ) {
          return true;
        }
      }
      return false;
    }

    EvaluatorInfo createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const {
      EvaluatorInfo leftEvaluatorInfo ;
      if( iBegin == iEnd) {
        return leftEvaluatorInfo;
      }
      //Start with '+'
      if (*iBegin == '+' and iEnd -iBegin > 1 and not std::isdigit( *(iBegin+1) ) ) {
        leftEvaluatorInfo = createEvaluator(iBegin+1, iEnd);

        //have to account for the '+' we skipped over
        leftEvaluatorInfo.nextParseIndex +=1;
        if( nullptr == leftEvaluatorInfo.evaluator.get() ) {
          return leftEvaluatorInfo;
        }
        if( leftEvaluatorInfo.nextParseIndex == iEnd-iBegin) {
          return leftEvaluatorInfo;
        }
      }
      //Start with '-'
      else if (*iBegin == '-' and iEnd -iBegin > 1 and not std::isdigit( *(iBegin+1) ) ) {
        leftEvaluatorInfo = createEvaluator(iBegin+1, iEnd);

        //have to account for the '+' we skipped over
        leftEvaluatorInfo.nextParseIndex +=1;
        if( nullptr == leftEvaluatorInfo.evaluator.get() ) {
          return leftEvaluatorInfo;
        }
        leftEvaluatorInfo.evaluator = std::unique_ptr<reco::formula::EvaluatorBase>( new reco::formula::UnaryMinusEvaluator( std::move(leftEvaluatorInfo.evaluator)) );
        if( leftEvaluatorInfo.nextParseIndex == iEnd-iBegin) {
          return leftEvaluatorInfo;
        }
      }
      //Start with '('
      else if( *iBegin == '(') {
        auto endParenthesis = findMatchingParenthesis(iBegin,iEnd);
        if(iBegin== endParenthesis) {
          return leftEvaluatorInfo;
        }
        leftEvaluatorInfo = createEvaluator(iBegin+1,endParenthesis);
        ++leftEvaluatorInfo.nextParseIndex;
        if(leftEvaluatorInfo.evaluator.get() == nullptr) {
          return leftEvaluatorInfo;
        }
        //need to account for closing parenthesis
        ++leftEvaluatorInfo.nextParseIndex;
        leftEvaluatorInfo.evaluator->setPrecidenceToParenthesis();
        if( iBegin+leftEvaluatorInfo.nextParseIndex == iEnd) {
          return leftEvaluatorInfo;
        }
      } else {
        //Does not start with a '('
        int maxParseDistance = 0;
        for( auto const& e: m_elements) {
          if(e->checkStart(*iBegin) ) {
            leftEvaluatorInfo = e->createEvaluator(iBegin,iEnd);
            if(leftEvaluatorInfo.evaluator != nullptr) {
              break;
            }
            if (leftEvaluatorInfo.nextParseIndex > maxParseDistance) {
              maxParseDistance = leftEvaluatorInfo.nextParseIndex;
            }
          }
        }
        if(leftEvaluatorInfo.evaluator.get() == nullptr) {
          //failed to parse
          leftEvaluatorInfo.nextParseIndex = maxParseDistance;
          return leftEvaluatorInfo;
        }
      }
      //did we evaluate the full expression?
      if(leftEvaluatorInfo.nextParseIndex == iEnd-iBegin) {
        return leftEvaluatorInfo;
      }

      //see if this is a binary expression
      auto fullExpression = createBinaryOperatorEvaluator(*this, std::move(leftEvaluatorInfo.evaluator), iBegin+leftEvaluatorInfo.nextParseIndex, iEnd);
      fullExpression.nextParseIndex +=leftEvaluatorInfo.nextParseIndex;
      fullExpression.maxNumVariables = std::max(leftEvaluatorInfo.maxNumVariables, fullExpression.maxNumVariables);
      fullExpression.maxNumParameters = std::max(leftEvaluatorInfo.maxNumParameters, fullExpression.maxNumParameters);
      if (iBegin + fullExpression.nextParseIndex != iEnd) {
        //did not parse the full expression
        fullExpression.evaluator.release();
      }
      return fullExpression;
    }

  private:
    std::vector<std::unique_ptr<ExpressionElementFinderBase>> m_elements;

  };

  template<typename Op>
  EvaluatorInfo createBinaryOperatorEvaluatorT(int iSymbolLength,
                                               reco::formula::EvaluatorBase::Precidence iPrec,
                                               ExpressionFinder const& iEF,
                                               std::unique_ptr<reco::formula::EvaluatorBase> iLHS,
                                               std::string::const_iterator iBegin,
                                               std::string::const_iterator iEnd) {
    EvaluatorInfo evalInfo = iEF.createEvaluator(iBegin+iSymbolLength,iEnd);
    evalInfo.nextParseIndex += iSymbolLength;

    if(evalInfo.evaluator.get() == nullptr) {
      return evalInfo;
    }

    if( static_cast<unsigned int>(iPrec) >= evalInfo.evaluator->precidence() ) {
      auto b = dynamic_cast<reco::formula::BinaryOperatorEvaluatorBase*>( evalInfo.evaluator.get() );
      assert(b != nullptr);
      std::unique_ptr<reco::formula::EvaluatorBase> temp;
      b->swapLeftEvaluator(temp);
      std::unique_ptr<reco::formula::EvaluatorBase> op{ new reco::formula::BinaryOperatorEvaluator<Op>(std::move(iLHS), std::move(temp), iPrec) };
      b->swapLeftEvaluator(op);

    } else {
      std::unique_ptr<reco::formula::EvaluatorBase> op{ new reco::formula::BinaryOperatorEvaluator<Op>(std::move(iLHS), std::move(evalInfo.evaluator), iPrec) };
        evalInfo.evaluator.swap(op);
    }
    return evalInfo;
  }

  struct power {
    double operator()(double iLHS, double iRHS) const {
      return std::pow(iLHS,iRHS);
    }
  };


  EvaluatorInfo 
  createBinaryOperatorEvaluator( ExpressionFinder const& iEF,
                                 std::unique_ptr<reco::formula::EvaluatorBase> iLHS,
                                 std::string::const_iterator iBegin,
                                 std::string::const_iterator iEnd) {
    EvaluatorInfo evalInfo;
    if(iBegin == iEnd) {
      return evalInfo;
    }

    if(*iBegin == '+') {
      return createBinaryOperatorEvaluatorT<std::plus<double>>(1,
                                                               reco::formula::EvaluatorBase::Precidence::kPlusMinus,
                                                               iEF,
                                                               std::move(iLHS),
                                                               iBegin,
                                                               iEnd);
    }

    else if(*iBegin == '-') {
      return createBinaryOperatorEvaluatorT<std::minus<double>>(1,
                                                                reco::formula::EvaluatorBase::Precidence::kPlusMinus,
                                                                iEF,
                                                                std::move(iLHS),
                                                                iBegin,
                                                                iEnd);
    }
    else if(*iBegin == '*') {
      return createBinaryOperatorEvaluatorT<std::multiplies<double>>(1,
                                                                     reco::formula::EvaluatorBase::Precidence::kMultDiv,
                                                                     iEF,
                                                                     std::move(iLHS),
                                                                     iBegin,
                                                                     iEnd);
    }
    else if(*iBegin == '/') {
      return createBinaryOperatorEvaluatorT<std::divides<double>>(1,
                                                                  reco::formula::EvaluatorBase::Precidence::kMultDiv,
                                                                  iEF,
                                                                  std::move(iLHS),
                                                                  iBegin,
                                                                  iEnd);
    }

    else if(*iBegin == '^') {
      return createBinaryOperatorEvaluatorT<power>(1,
                                                                  reco::formula::EvaluatorBase::Precidence::kMultDiv,
                                                                  iEF,
                                                                  std::move(iLHS),
                                                                  iBegin,
                                                                  iEnd);
    }

    return evalInfo;
  }


  template<typename Op>
  EvaluatorInfo
  checkForSingleArgFunction(std::string::const_iterator iBegin,
                            std::string::const_iterator iEnd,
                            ExpressionFinder const* iExpressionFinder,
                            const std::string& iName,
                            Op op) {
    EvaluatorInfo info;
    if(iName.size()+2 > static_cast<unsigned int>(iEnd-iBegin) ) {
      return info;
    }
    auto pos = iName.find(&(*iBegin), 0,iName.size());

    if(std::string::npos == pos or *(iBegin+iName.size()) != '(') {
      return info;
    }
    
    info.nextParseIndex = iName.size()+1;

    auto itEndParen = findMatchingParenthesis(iBegin+iName.size(),iEnd);
    if(iBegin+iName.size() == itEndParen) {
      return info;
    }

    auto argEvaluatorInfo = iExpressionFinder->createEvaluator(iBegin+iName.size()+1, itEndParen);
    info.nextParseIndex += argEvaluatorInfo.nextParseIndex;
    if(argEvaluatorInfo.evaluator.get() == nullptr or info.nextParseIndex+1 != 1+itEndParen - iBegin) {
      return info;
    }
    //account for closing parenthesis
    ++info.nextParseIndex;

    info.evaluator.reset( new reco::formula::FunctionOneArgEvaluator(std::move(argEvaluatorInfo.evaluator),
                                                                     op) );
    return info;
  }

  std::string::const_iterator findCommaNotInParenthesis(std::string::const_iterator iBegin,
                                                        std::string::const_iterator iEnd ) {
    int level = 0;
    std::string::const_iterator it = iBegin;
    for(; it != iEnd; ++it) {
      if (*it == '(') {
        ++level;
      } else if(*it == ')') {
        --level;
      }
      else if( *it ==',' and level == 0 ) {
        return it;
      }
    }

    return it;
  }


  template<typename Op>
  EvaluatorInfo
  checkForTwoArgsFunction(std::string::const_iterator iBegin,
                          std::string::const_iterator iEnd,
                          ExpressionFinder const* iExpressionFinder,
                          const std::string& iName,
                          Op op) {
    EvaluatorInfo info;
    if(iName.size()+2 > static_cast<unsigned int>(iEnd-iBegin) ) {
      return info;
    }
    auto pos = iName.find(&(*iBegin), 0,iName.size());

    if(std::string::npos == pos or *(iBegin+iName.size()) != '(') {
      return info;
    }
    
    info.nextParseIndex = iName.size()+1;

    auto itEndParen = findMatchingParenthesis(iBegin+iName.size(),iEnd);
    if(iBegin+iName.size() == itEndParen) {
      return info;
    }

    auto itComma = findCommaNotInParenthesis(iBegin+iName.size()+1, itEndParen);

    auto arg1EvaluatorInfo = iExpressionFinder->createEvaluator(iBegin+iName.size()+1, itComma);
    info.nextParseIndex += arg1EvaluatorInfo.nextParseIndex;
    if(arg1EvaluatorInfo.evaluator.get() == nullptr or info.nextParseIndex != itComma-iBegin ) {
      return info;
    }
    //account for commas
    ++info.nextParseIndex;

    auto arg2EvaluatorInfo = iExpressionFinder->createEvaluator(itComma+1, itEndParen);
    info.nextParseIndex += arg2EvaluatorInfo.nextParseIndex;

    if(arg2EvaluatorInfo.evaluator.get() == nullptr or info.nextParseIndex+1 != 1+itEndParen - iBegin) {
      return info;
    }
    //account for closing parenthesis
    ++info.nextParseIndex;

    info.evaluator.reset( new reco::formula::FunctionTwoArgsEvaluator(std::move(arg1EvaluatorInfo.evaluator),
                                                                      std::move(arg2EvaluatorInfo.evaluator),
                                                                      op) );
    return info;
  }

  static const std::string k_log("log");
  static const std::string k_log10("log10");
  static const std::string k_TMath__Log("TMath::Log");
  double const kLog10Inv = 1./std::log(10.);
  static const std::string k_exp("exp");
  static const std::string k_max("max");


  EvaluatorInfo 
  FunctionFinder::createEvaluator(std::string::const_iterator iBegin, std::string::const_iterator iEnd) const {
    EvaluatorInfo info;

    info = checkForSingleArgFunction(iBegin, iEnd, m_expressionFinder,
                                     k_log, [](double iArg)->double { return std::log(iArg); } );
    if(info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(iBegin, iEnd, m_expressionFinder,
                                     k_TMath__Log, [](double iArg)->double { return std::log(iArg); } );
    if(info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(iBegin, iEnd, m_expressionFinder,
                                     k_log10, [](double iArg)->double { return std::log(iArg)*kLog10Inv; } );
    if(info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForSingleArgFunction(iBegin, iEnd, m_expressionFinder,
                                     k_exp, [](double iArg)->double { return std::exp(iArg); } );
    if(info.evaluator.get() != nullptr) {
      return info;
    }

    info = checkForTwoArgsFunction(iBegin, iEnd, m_expressionFinder,
                                   k_max, [](double iArg1, double iArg2)->double { return std::max(iArg1,iArg2); } );
    if(info.evaluator.get() != nullptr) {
      return info;
    }

    return info;
  };

  static ExpressionFinder const s_expressionFinder;
  
}
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FormulaEvaluator::FormulaEvaluator( std::string const& iFormula )
{
  auto info  = s_expressionFinder.createEvaluator(iFormula.begin(), iFormula.end());

  if(info.nextParseIndex != static_cast<int>(iFormula.size()) or info.evaluator.get() == nullptr) {
    throw cms::Exception("FormulaEvaluatorParseError")<<"While parsing '"<<iFormula<<"' could not parse beyond '"<<std::string(iFormula.begin(),iFormula.begin()+info.nextParseIndex) <<"'";
  }
  m_evaluator = std::move(info.evaluator);
  m_nVariables = info.maxNumVariables;
  m_nParameters = info.maxNumParameters;
}

//
// const member functions
//
double
FormulaEvaluator::evaluate(double const* iVariables, double const* iParameters) const
{
  return m_evaluator->evaluate(iVariables, iParameters);
}

void 
FormulaEvaluator::throwWrongNumberOfVariables(size_t iSize) const {
  throw cms::Exception("WrongNumVariables")<<"FormulaEvaluator expected at least "<<m_nVariables<<" but was passed only "<<iSize;
}
void 
FormulaEvaluator::throwWrongNumberOfParameters(size_t iSize) const {
  throw cms::Exception("WrongNumParameters")<<"FormulaEvaluator expected at least "<<m_nParameters<<" but was passed only "<<iSize;
}

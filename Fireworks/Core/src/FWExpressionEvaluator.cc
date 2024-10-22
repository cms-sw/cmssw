// -*- C++ -*-
//
// Package:     Core
// Class  :     FWExpressionEvaluator
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 29 13:39:56 PST 2008
//

// system include files
#include <sstream>

#include "FWCore/Reflection/interface/ObjectWithDict.h"

// user include files
#include "Fireworks/Core/interface/FWExpressionEvaluator.h"
#include "Fireworks/Core/interface/FWExpressionException.h"
#include "CommonTools/Utils/interface/parser/Grammar.h"
#include "CommonTools/Utils/interface/parser/Exception.h"

#include "Fireworks/Core/src/expressionFormatHelpers.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWExpressionEvaluator::FWExpressionEvaluator(const std::string& iExpression, const std::string& iClassName)
    : m_className(iClassName), m_type(edm::TypeWithDict::byName(iClassName)) {
  setExpression(iExpression);
}

// FWExpressionEvaluator::FWExpressionEvaluator(const FWExpressionEvaluator& rhs)
// {
//    // do actual copying here;
// }

FWExpressionEvaluator::~FWExpressionEvaluator() {}

//
// assignment operators
//
// const FWExpressionEvaluator& FWExpressionEvaluator::operator=(const FWExpressionEvaluator& rhs)
// {
//   //An exception safe implementation is
//   FWExpressionEvaluator temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWExpressionEvaluator::setExpression(const std::string& iExpression) {
  if (m_type != edm::TypeWithDict() && !iExpression.empty()) {
    using namespace fireworks::expression;

    //Backwards compatibility with old format
    std::string temp = oldToNewFormat(iExpression);

    //now setup the parser
    using namespace boost::spirit::classic;
    reco::parser::ExpressionPtr tmpPtr;
    reco::parser::Grammar grammar(tmpPtr, m_type);
    try {
      if (parse(temp.c_str(), grammar.use_parser<1>() >> end_p, space_p).full) {
        m_expr = tmpPtr;
        m_expression = iExpression;
      } else {
        throw FWExpressionException("syntax error", -1);
        //std::cout <<"failed to parse "<<iExpression<<" because of syntax error"<<std::endl;
      }
    } catch (const reco::parser::BaseException& e) {
      //NOTE: need to calculate actual position before doing the regex
      throw FWExpressionException(reco::parser::baseExceptionWhat(e),
                                  indexFromNewFormatToOldFormat(temp, e.where - temp.c_str(), iExpression));
      //std::cout <<"failed to parse "<<iExpression<<" because "<<reco::parser::baseExceptionWhat(e)<<std::endl;
    }
  } else {
    m_expression = iExpression;
  }
}

void FWExpressionEvaluator::setClassName(const std::string& iClassName) {
  //NOTE: How do we handle the case where the filter was created before
  // the library for the class was loaded and therefore we don't have
  // a dictionary for it?

  m_className = iClassName;
  m_type = edm::TypeWithDict::byName(iClassName);
  setExpression(m_expression);
}

//
// const member functions
//
const std::string& FWExpressionEvaluator::expression() const { return m_expression; }

double FWExpressionEvaluator::evalExpression(const void* iObject) const {
  if (m_expression.empty() || !m_expr.get()) {
    return 0;
  }

  edm::ObjectWithDict o(m_type, const_cast<void*>(iObject));
  return m_expr->value(o);
}

//
// static member functions
//

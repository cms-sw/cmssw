// -*- C++ -*-
#ifndef Fireworks_Core_FWExpressionEvaluator_h
#define Fireworks_Core_FWExpressionEvaluator_h
//
// Package:     Core
// Class  :     FWExpressionEvaluator
//
/**\class FWExpressionEvaluator FWExpressionEvaluator.h Fireworks/Core/interface/FWExpressionEvaluator.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 29 13:39:51 PST 2008
//

// system include files
#include <string>
#include "FWCore/Reflection/interface/TypeWithDict.h"

// user include files
#include "CommonTools/Utils/interface/parser/ExpressionPtr.h"
#include "CommonTools/Utils/interface/parser/ExpressionBase.h"

// forward declarations

class FWExpressionEvaluator {
public:
  FWExpressionEvaluator(const std::string& iExpression, const std::string& iClassName);
  virtual ~FWExpressionEvaluator();

  // ---------- const member functions ---------------------

  const std::string& expression() const;

  double evalExpression(const void*) const;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  /** Throws an FWExpressionException if there is a problem */
  void setExpression(const std::string&);
  void setClassName(const std::string&);

private:
  //FWExpressionEvaluator(const FWExpressionEvaluator&); // stop default

  //const FWExpressionEvaluator& operator=(const FWExpressionEvaluator&); // stop default

  // ---------- member data --------------------------------
  std::string m_expression;
  std::string m_className;
  reco::parser::ExpressionPtr m_expr;
  edm::TypeWithDict m_type;
};

#endif

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
// $Id: FWExpressionEvaluator.h,v 1.4 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include <string>
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files
#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/ExpressionBase.h"


// forward declarations

class FWExpressionEvaluator {

public:
   FWExpressionEvaluator(const std::string& iExpression,
			 const std::string& iClassName);
   virtual ~FWExpressionEvaluator();

   // ---------- const member functions ---------------------

   const std::string& expression() const;

   double evalExpression(const void*) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   /** Throws an FWExpressionException if there is a problem */
   void setExpression(const std::string& );
   void setClassName(const std::string& );

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

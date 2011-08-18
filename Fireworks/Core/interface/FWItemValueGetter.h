#ifndef Fireworks_Core_FWItemValueGetter_h
#define Fireworks_Core_FWItemValueGetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemValueGetter
//
/**\class FWItemValueGetter FWItemValueGetter.h Fireworks/Core/interface/FWItemValueGetter.h

   Description: Retrieves a particular value from an item

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 30 16:14:58 EST 2008
// $Id: FWItemValueGetter.h,v 1.3 2010/08/18 10:30:10 amraktad Exp $
//

#include <string>
#include <vector>
#include "Reflex/Member.h"
#include "Reflex/Type.h"

#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/ExpressionBase.h"


class FWItemValueGetter {

public:
   FWItemValueGetter(const ROOT::Reflex::Type&,
                     const std::vector<std::pair<std::string, std::string> >& iFindValueFrom);
   //virtual ~FWItemValueGetter();

   // ---------- const member functions ---------------------
   double valueFor(const void*) const;
   const std::string& stringValueFor(const void*) const;

   bool isValid() const;

   std::string valueName() const;
   const std::string& unit() const;

   void setValueAndUnit(const std::string& iValue, const std::string& iUnit);

private:
   //FWItemValueGetter(const FWItemValueGetter&); // stop default
   //const FWItemValueGetter& operator=(const FWItemValueGetter&); // stop default

   // ---------- member data --------------------------------
    
   ROOT::Reflex::Type m_type;
   // ROOT::Reflex::Member m_memberFunction;

   std::string m_expression;
   reco::parser::ExpressionPtr m_expr;
   std::string m_unit;
};

#endif

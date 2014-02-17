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
// $Id: FWItemValueGetter.h,v 1.8 2012/08/28 22:25:42 wmtan Exp $
//

#include <string>
#include <vector>

#include "Rtypes.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/ExpressionBase.h"


class FWItemValueGetter
{
public:
   FWItemValueGetter(const edm::TypeWithDict&, const std::string& iPurpose);
   double valueFor(const void*, int idx) const;
   UInt_t precision( int idx) const;
   std::vector<std::string> getTitles() const;
   int numValues() const; 

   const std::string& getToolTip(const void* iObject) const;


private:
   struct Entry {
      reco::parser::ExpressionPtr m_expr;
      std::string m_expression; 
      std::string m_unit;
      std::string m_title;
      UInt_t     m_precision;

      Entry(reco::parser::ExpressionPtr iExpr, std::string iExpression, std::string iUnit, std::string iTitle, int iPrec) :
         m_expr(iExpr), m_expression(iExpression), m_unit(iUnit), m_title(iTitle), m_precision(iPrec) {}
   };

   bool addEntry(std::string iExpression,int iPrec = 2,  std::string iTitle = "",  std::string iUnit = "");

   typedef std::vector<Entry > Entries_t;
   Entries_t::const_iterator Entries_i;

   Entries_t m_entries;
   edm::TypeWithDict m_type;

   int m_titleWidth;

   
};

#endif

// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemValueGetter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 30 16:15:43 EST 2008
// $Id: FWItemValueGetter.cc,v 1.6 2010/09/06 12:35:59 elmer Exp $
//

// system include files
#include <sstream>
#include "Reflex/Object.h"
#include "Reflex/Base.h"
#include <cstdio>

// user include files
#include "Fireworks/Core/interface/FWItemValueGetter.h"

#include "Fireworks/Core/interface/FWExpressionEvaluator.h"
#include "Fireworks/Core/interface/FWExpressionException.h"
#include "CommonTools/Utils/src/Grammar.h"
#include "CommonTools/Utils/interface/Exception.h"

#include "Fireworks/Core/src/expressionFormatHelpers.h"

FWItemValueGetter::FWItemValueGetter(const ROOT::Reflex::Type& iType,
                                     const std::vector<std::pair<std::string, std::string> >& iFindValueFrom) :
   m_type(iType)
{
   using namespace boost::spirit::classic;
   reco::parser::ExpressionPtr tmpPtr;
   reco::parser::Grammar grammar(tmpPtr,m_type);

   for(std::vector<std::pair<std::string,std::string> >::const_iterator it = iFindValueFrom.begin(); it !=  iFindValueFrom.end(); ++it)
   {
      const std::string& iExpression = it->first;
      if(m_type != ROOT::Reflex::Type() && iExpression.size()) {
  
         using namespace fireworks::expression;

         //Backwards compatibility with old format
         std::string temp = oldToNewFormat(iExpression);

         //now setup the parser
         try {
            if(parse(temp.c_str(), grammar.use_parser<1>() >> end_p, space_p).full) {
               m_expr = tmpPtr;
               m_expression = iExpression;
            } else {
               throw FWExpressionException("syntax error", -1);
               // std::cout <<"failed to parse "<<iExpression<<" because of syntax error"<<std::endl;
            }

            m_expression=iExpression;
            m_unit = it->second;
            break;

         } 
         catch(const reco::parser::BaseException& e) {
              // std::cout <<"failed to parse "<<iExpression<<" because "<<reco::parser::baseExceptionWhat(e)<<std::endl;
         }
      }
   }
}


double
FWItemValueGetter::valueFor(const void* iObject) const
{
   if(m_expression.empty() || !m_expr.get()) {
      return 0;
   }

   ROOT::Reflex::Object o(m_type, const_cast<void *>(iObject));
   return m_expr->value(o);
}

const std::string&
FWItemValueGetter::stringValueFor(const void* iObject) const
{ 
   static std::string buff(128, 0);
   double v = valueFor(iObject);
   snprintf(&buff[0], 127, "%.1f %s", v, m_unit.c_str());
   return buff;
}

bool
FWItemValueGetter::isValid() const
{
   return  !m_expression.empty();
}

std::string
FWItemValueGetter::valueName() const
{
   return  m_expression;
}

const std::string&
FWItemValueGetter::unit() const
{
   return m_unit;
}

//
// static member functions
//

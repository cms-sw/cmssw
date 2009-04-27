// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelExpressionSelector
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Jan 23 10:37:22 EST 2008
// $Id: FWModelExpressionSelector.cc,v 1.8.8.1 2009/04/24 02:18:41 dmytro Exp $
//

// system include files
#include <sstream>
#include "TClass.h"
#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "CommonTools/Utils/src/Grammar.h"
#include "CommonTools/Utils/interface/Exception.h"

// user include files
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/src/fwCintInterfaces.h"
#include "Fireworks/Core/interface/FWExpressionException.h"
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
/*
   FWModelExpressionSelector::FWModelExpressionSelector()
   {
   }
 */
// FWModelExpressionSelector::FWModelExpressionSelector(const FWModelExpressionSelector& rhs)
// {
//    // do actual copying here;
// }

/*FWModelExpressionSelector::~FWModelExpressionSelector()
   {
   }
 */
//
// assignment operators
//
// const FWModelExpressionSelector& FWModelExpressionSelector::operator=(const FWModelExpressionSelector& rhs)
// {
//   //An exception safe implementation is
//   FWModelExpressionSelector temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
bool
FWModelExpressionSelector::select(FWEventItem* iItem, const std::string& iExpression) const
{
   using namespace fireworks::expression;

   ROOT::Reflex::Type type= ROOT::Reflex::Type::ByName(iItem->modelType()->GetName());
   assert(type != ROOT::Reflex::Type());

   //Backwards compatibility with old format
   std::string temp = oldToNewFormat(iExpression);

   //now setup the parser
   using namespace boost::spirit::classic;
   reco::parser::SelectorPtr selectorPtr;
   reco::parser::Grammar grammar(selectorPtr,type);
   bool succeeded=true;
   try {
      if(!parse(temp.c_str(), grammar.use_parser<0>() >> end_p, space_p).full) {
         throw FWExpressionException("syntax error", -1);
         //std::cout <<"failed to parse "<<iExpression<<" because of syntax error"<<std::endl;
         succeeded=false;
      }
   } catch(const reco::parser::BaseException& e) {
      //NOTE: need to calculate actual position before doing the regex
      throw FWExpressionException(reco::parser::baseExceptionWhat(e), indexFromNewFormatToOldFormat(temp,e.where-temp.c_str(),iExpression));
      //std::cout <<"failed to parse "<<iExpression<<" because "<<reco::parser::baseExceptionWhat(e)<<std::endl;
      succeeded=false;
   }
   if(!succeeded) { return false;}


   FWChangeSentry sentry(*(iItem->changeManager()));
   for( unsigned int index = 0; index < iItem->size(); ++index ) {
      ROOT::Reflex::Object o(type, const_cast<void *>(iItem->modelData(index)));

      if((*selectorPtr)(o)) {
         iItem->select(index);
      }
   }

   return true;
}

//
// static member functions
//

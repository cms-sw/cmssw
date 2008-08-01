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
// $Id: FWModelExpressionSelector.cc,v 1.2 2008/06/12 20:29:59 chrjones Exp $
//

// system include files
#include <sstream>
#include <boost/regex.hpp>
#include "TClass.h"
#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "PhysicsTools/Utilities/src/Grammar.h"
#include "FWCore/Utilities/interface/EDMException.h"

// user include files
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/src/fwCintInterfaces.h"


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
   ROOT::Reflex::Type type= ROOT::Reflex::Type::ByName(iItem->modelType()->GetName());
   assert(type != ROOT::Reflex::Type());

   //Backwards compatibility with old format: If find a $. or a () just remove them
   const std::string variable;
   static boost::regex const reVarName("(\\$\\.)|(\\(\\))");
   
   std::string temp = boost::regex_replace(iExpression,reVarName,variable);

   //now setup the parser
   using namespace boost::spirit;
   reco::parser::SelectorPtr selectorPtr;
   reco::parser::Grammar grammar(selectorPtr,type);
   bool succeeded=true;
   try {
      if(!parse(temp.c_str(), grammar.use_parser<0>() >> end_p, space_p).full) {
         std::cout <<"failed to parse "<<iExpression<<std::endl;
         succeeded=false;
      }
   }catch(const edm::Exception& e) {
      std::cout <<"failed to parse "<<iExpression<<" because "<<e.what()<<std::endl;
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

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
// $Id: FWModelExpressionSelector.cc,v 1.1 2008/01/24 00:30:30 chrjones Exp $
//

// system include files
#include <sstream>
#include <boost/regex.hpp>

#include "TROOT.h"
#include "TClass.h"
#include "TInterpreter.h"

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
   const std::string variable(std::string("(*((const ")+iItem->modelType()->GetName()+"*)(fwGetObjectPtr())))");
   static boost::regex const reVarName("\\$");
   
   std::string fullExpression(std::string("fwSetInCint((long)(")+iExpression+"))");
   
   fullExpression = boost::regex_replace(fullExpression,reVarName,variable);
   
   
   /*
   const std::string modelPrefix= std::string("(*((const ")+iItem->modelType()->GetName()+"*)(";
   const std::string modelPostfix(")))");
   
   const std::string expression(std::string("(Long_t)(")+iExpression+")");
   
   static boost::regex const reVarName("\\$");
    */
   bool returnValue = true;

   //we are probably making many changes
   FWChangeSentry sentry(*(iItem->changeManager()));
   for( unsigned int index = 0; index < iItem->size(); ++index ) {
      fwSetObjectPtr(iItem->modelData(index));
      fwCintReturnType()=kFWCintReturnNoReturn;
      Int_t error = 0;
      gROOT->ProcessLineFast(fullExpression.c_str(),
                             &error);
      
      /*
      std::stringstream fullVariable;
      fullVariable <<modelPrefix<<iItem->modelData(index)<<modelPostfix;
      Int_t error = 0;
      Long_t value = gROOT->ProcessLineFast(boost::regex_replace(expression,reVarName,fullVariable.str()).c_str(),
                                            &error);
       */
      if(TInterpreter::kNoError != error || fwCintReturnType() == kFWCintReturnNoReturn) {
         if(index==0) {
            return false;
         }
         returnValue = false;
      } else {
         if(fwGetFromCintLong()) {
            iItem->select(index);
         }
      }
   }
   return returnValue;
}

//
// static member functions
//

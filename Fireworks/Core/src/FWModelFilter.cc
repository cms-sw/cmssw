// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelFilter
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 29 13:39:56 PST 2008
// $Id: FWModelFilter.cc,v 1.1 2008/03/01 02:14:08 chrjones Exp $
//

// system include files
#include <sstream>
#include <boost/regex.hpp>

#include "TROOT.h"
#include "TClass.h"
#include "TInterpreter.h"

// user include files
#include "Fireworks/Core/interface/FWModelFilter.h"
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
FWModelFilter::FWModelFilter(const std::string& iExpression,
                           const std::string& iClassName):
m_className(iClassName)
{
   setExpression(iExpression);
}

// FWModelFilter::FWModelFilter(const FWModelFilter& rhs)
// {
//    // do actual copying here;
// }

FWModelFilter::~FWModelFilter()
{
}

//
// assignment operators
//
// const FWModelFilter& FWModelFilter::operator=(const FWModelFilter& rhs)
// {
//   //An exception safe implementation is
//   FWModelFilter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWModelFilter::setExpression(const std::string& iExpression)
{
   m_expression = iExpression;
   const std::string variable(std::string("(*((const ")+m_className+"*)(fwGetObjectPtr())))");
   static boost::regex const reVarName("\\$");

   std::string temp(std::string("fwSetInCint((long)(")+iExpression+"))");

   temp = boost::regex_replace(temp,reVarName,variable);
   m_fullExpression.swap(temp);
}

void 
FWModelFilter::setClassName(const std::string& iClassName)
{
   m_className = iClassName;
   setExpression(m_expression);
   //std::string temp(std::string("(*((const ")+iClassName+"*)(");
   //m_prefix.swap(temp);
}

//
// const member functions
//
const std::string& 
FWModelFilter::expression() const
{
   return m_expression;
}

bool 
FWModelFilter::passesFilter(const void* iObject) const
{
   bool returnValue=true;
   if(m_expression.empty()) {
      return true;
   }
   
   fwSetObjectPtr(iObject);
   fwCintReturnType()=kFWCintReturnNoReturn;
   Int_t error = 0;
   Long_t value = gROOT->ProcessLineFast(m_fullExpression.c_str(),
                                         &error);
   if(TInterpreter::kNoError != error || fwCintReturnType() == kFWCintReturnNoReturn) {
      returnValue = true;
   } else {
      returnValue = fwGetFromCintLong();
   }
   return returnValue;
}

const bool 
FWModelFilter::trivialFilter() const
{
   return m_expression.empty();
}

//
// static member functions
//

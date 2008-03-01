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
// $Id$
//

// system include files
#include <sstream>
#include <boost/regex.hpp>

#include "TROOT.h"
#include "TClass.h"
#include "TInterpreter.h"

// user include files
#include "Fireworks/Core/interface/FWModelFilter.h"


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
m_expression(iExpression),
m_className(iClassName),
m_prefix(std::string("(*((const ")+iClassName+"*)("),
m_castExpression(std::string("(Long_t)(")+iExpression+")")
{

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
   std::string temp(std::string("(Long_t)(")+iExpression+")");
   m_castExpression.swap(temp);
}

void 
FWModelFilter::setClassName(const std::string& iClassName)
{
   m_className = iClassName;
   std::string temp(std::string("(*((const ")+iClassName+"*)(");
   m_prefix.swap(temp);
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
   if(m_expression.empty()) {
      return true;
   }
   bool returnValue = true;
   static const std::string s_postfix(")))");
   static boost::regex const reVarName("\\$");

   //we are probably making many changes
   std::stringstream fullVariable;
   fullVariable <<m_prefix<<iObject<<s_postfix;
   Int_t error = 0;
   Long_t value = gROOT->ProcessLineFast(boost::regex_replace(m_castExpression,reVarName,fullVariable.str()).c_str(),
                                         &error);
   if(TInterpreter::kNoError != error) {
      returnValue = true;
   } else {
      returnValue = value;
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

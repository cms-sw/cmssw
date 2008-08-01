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
// $Id: FWModelFilter.cc,v 1.4 2008/08/01 13:43:15 chrjones Exp $
//

// system include files
#include <sstream>
#include <boost/regex.hpp>

#include "TROOT.h"
#include "TClass.h"
#include "TInterpreter.h"
#include "Reflex/Object.h"

// user include files
#include "Fireworks/Core/interface/FWModelFilter.h"
#include "Fireworks/Core/src/fwCintInterfaces.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/src/Grammar.h"

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
m_className(iClassName),
m_type(ROOT::Reflex::Type::ByName(iClassName))
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
   if(m_type != ROOT::Reflex::Type() && iExpression.size()) {
      
      //Backwards compatibility with old format: If find a $. or a () just remove them
      const std::string variable;
      static boost::regex const reVarName("(\\$\\.)|(\\(\\))");
      
      std::string temp = boost::regex_replace(iExpression,reVarName,variable);
      
      //now setup the parser
      using namespace boost::spirit;
      reco::parser::SelectorPtr tmpPtr;
      reco::parser::Grammar grammar(tmpPtr,m_type);
      try {
         if(parse(temp.c_str(), grammar.use_parser<0>() >> end_p, space_p).full) {
            m_selector = tmpPtr;
            m_expression = iExpression;
         } else {
            std::cout <<"failed to parse "<<iExpression<<std::endl;
         }
      }catch(const edm::Exception& e) {
         std::cout <<"failed to parse "<<iExpression<<" because "<<e.what()<<std::endl;
      }
   }else {
      m_expression=iExpression;
   }
}

void 
FWModelFilter::setClassName(const std::string& iClassName)
{
   //NOTE: How do we handle the case where the filter was created before
   // the library for the class was loaded and therefore we don't have
   // a Reflex dictionary for it?
   
   m_className = iClassName;
   m_type = ROOT::Reflex::Type::ByName(iClassName);
   setExpression(m_expression);
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
   if(m_expression.empty() || !m_selector.get()) {
      return true;
   }
   
   ROOT::Reflex::Object o(m_type, const_cast<void *>(iObject));
   return (*m_selector)(o);  
}

const bool 
FWModelFilter::trivialFilter() const
{
   return m_expression.empty();
}

//
// static member functions
//

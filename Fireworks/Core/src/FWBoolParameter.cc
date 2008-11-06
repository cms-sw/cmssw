// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoolParameter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:41 EST 2008
// $Id: FWBoolParameter.cc,v 1.2 2008/09/27 16:55:02 dmytro Exp $
//

// system include files
#include <assert.h>
#include <sstream>

// user include files
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWConfiguration.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWBoolParameter::FWBoolParameter(FWParameterizable* iParent,
				 const std::string& iName,
				 bool iDefault):
FWParameterBase(iParent,iName),
m_value(iDefault)
{
}

// FWBoolParameter::FWBoolParameter(const FWBoolParameter& rhs)
// {
//    // do actual copying here;
// }
/*
FWBoolParameter::~FWBoolParameter()
{
}
*/
//
// assignment operators
//
// const FWBoolParameter& FWBoolParameter::operator=(const FWBoolParameter& rhs)
// {
//   //An exception safe implementation is
//   FWBoolParameter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWBoolParameter::setFrom(const FWConfiguration& iFrom)
{
   if (const FWConfiguration* config = iFrom.valueForKey(name()) ) {
      std::istringstream s(config->value());
      s>>m_value;
   }
   changed_(m_value);
}

void
FWBoolParameter::set(bool iValue)
{
   m_value = iValue;
   changed_(iValue);
}

//
// const member functions
//
void
FWBoolParameter::addTo(FWConfiguration& iTo) const
{
   std::ostringstream s;
   s<<m_value;
   iTo.addKeyValue(name(),FWConfiguration(s.str()));
}

//
// static member functions
//

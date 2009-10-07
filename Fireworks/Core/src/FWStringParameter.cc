// -*- C++ -*-
//
// Package:     Core
// Class  :     FWStringParameter
// $Id: FWStringParameter.cc,v 1.4 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <assert.h>
#include <sstream>

// user include files
#include "Fireworks/Core/interface/FWStringParameter.h"
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
FWStringParameter::FWStringParameter(FWParameterizable* iParent,
				     const std::string& iName,
				     const std::string& iDefault) :
   FWParameterBase(iParent,iName),
   m_value(iDefault)
{
}

void
FWStringParameter::setFrom(const FWConfiguration& iFrom)
{
   if (const FWConfiguration* config = iFrom.valueForKey(name()) ) {
      std::istringstream s(config->value());
      s>>m_value;
   }
   changed_(m_value);
}

void
FWStringParameter::set(const std::string& iValue)
{
   m_value = iValue;
   changed_(iValue);
}

//
// const member functions
//
void
FWStringParameter::addTo(FWConfiguration& iTo) const
{
   std::ostringstream s;
   s<<m_value;
   iTo.addKeyValue(name(),FWConfiguration(s.str()));
}

//
// static member functions
//

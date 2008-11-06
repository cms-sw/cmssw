// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDoubleParameter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:41 EST 2008
// $Id: FWDoubleParameter.cc,v 1.2 2008/09/27 16:55:02 dmytro Exp $
//

// system include files
#include <assert.h>
#include <sstream>

// user include files
#include "Fireworks/Core/interface/FWDoubleParameter.h"
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
FWDoubleParameter::FWDoubleParameter(FWParameterizable* iParent,
                                     const std::string& iName,
                                     double iDefault,
                                     double iMin,
                                     double iMax):
FWParameterBase(iParent,iName),
m_value(iDefault),
m_min(iMin),
m_max(iMax)
{
}

// FWDoubleParameter::FWDoubleParameter(const FWDoubleParameter& rhs)
// {
//    // do actual copying here;
// }
/*
FWDoubleParameter::~FWDoubleParameter()
{
}
*/
//
// assignment operators
//
// const FWDoubleParameter& FWDoubleParameter::operator=(const FWDoubleParameter& rhs)
// {
//   //An exception safe implementation is
//   FWDoubleParameter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWDoubleParameter::setFrom(const FWConfiguration& iFrom)
{
   if (const FWConfiguration* config = iFrom.valueForKey(name()) ) {
      std::istringstream s(config->value());
      s>>m_value;
   }
   changed_(m_value);
}

void
FWDoubleParameter::set(double iValue)
{
   m_value = iValue;
   changed_(iValue);
}

//
// const member functions
//
void
FWDoubleParameter::addTo(FWConfiguration& iTo) const
{
   std::ostringstream s;
   s<<m_value;
   iTo.addKeyValue(name(),FWConfiguration(s.str()));
}

//
// static member functions
//

// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfigurableParameterizable
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Mar 16 12:01:36 EDT 2008
// $Id: FWConfigurableParameterizable.cc,v 1.5 2012/09/21 09:26:26 eulisse Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include <cassert>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWConfigurableParameterizable::FWConfigurableParameterizable(unsigned int iVersion) :
   m_version(iVersion)
{
}

// FWConfigurableParameterizable::FWConfigurableParameterizable(const FWConfigurableParameterizable& rhs)
// {
//    // do actual copying here;
// }

FWConfigurableParameterizable::~FWConfigurableParameterizable()
{
}

//
// assignment operators
//
// const FWConfigurableParameterizable& FWConfigurableParameterizable::operator=(const FWConfigurableParameterizable& rhs)
// {
//   //An exception safe implementation is
//   FWConfigurableParameterizable temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWConfigurableParameterizable::setFrom(const FWConfiguration& iFrom)
{
   //need a way to handle versioning
   assert(iFrom.version() == m_version);
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);
   }
}

//
// const member functions
//
void
FWConfigurableParameterizable::addTo(FWConfiguration& oTo) const
{
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->addTo(oTo);
   }
}

//
// static member functions
//

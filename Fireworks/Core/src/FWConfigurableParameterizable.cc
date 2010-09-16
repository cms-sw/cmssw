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
// $Id: FWConfigurableParameterizable.cc,v 1.2 2008/11/06 22:05:25 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWParameterBase.h"
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

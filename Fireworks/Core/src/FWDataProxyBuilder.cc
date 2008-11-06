// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDataProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FWDataProxyBuilder.cc,v 1.3 2008/06/09 19:54:03 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWDataProxyBuilder.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWDataProxyBuilder::FWDataProxyBuilder():
  m_item(0)
{
}

// FWDataProxyBuilder::FWDataProxyBuilder(const FWDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWDataProxyBuilder::~FWDataProxyBuilder()
{
}

//
// assignment operators
//
// const FWDataProxyBuilder& FWDataProxyBuilder::operator=(const FWDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
}

void
FWDataProxyBuilder::build(TObject** iObject)
{
  if(0!= m_item) {
    build(m_item, iObject);
  }
}
//
// const member functions
//
const std::string
FWDataProxyBuilder::purpose() const
{
   return std::string();
}

//
// static member functions
//

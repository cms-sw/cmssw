// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FWRPZDataProxyBuilder.cc,v 1.1 2007/12/09 22:49:23 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWRPZDataProxyBuilder::FWRPZDataProxyBuilder():
  m_item(0)
{
}

// FWRPZDataProxyBuilder::FWRPZDataProxyBuilder(const FWRPZDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRPZDataProxyBuilder::~FWRPZDataProxyBuilder()
{
}

//
// assignment operators
//
// const FWRPZDataProxyBuilder& FWRPZDataProxyBuilder::operator=(const FWRPZDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWRPZDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWRPZDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
}

void
FWRPZDataProxyBuilder::build(TEveElementList** iObject)
{
  if(0!= m_item) {
    build(m_item, iObject);
  }
}
//
// const member functions
//

//
// static member functions
//

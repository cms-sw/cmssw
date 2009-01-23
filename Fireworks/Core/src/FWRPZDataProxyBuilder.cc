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
// $Id: FWRPZDataProxyBuilder.cc,v 1.20 2008/11/06 22:05:26 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"

#include "Fireworks/Core/src/changeElementAndChildren.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
TEveCalo3D* FWRPZDataProxyBuilder::m_calo3d = 0;

//
// constructors and destructor
//
FWRPZDataProxyBuilder::FWRPZDataProxyBuilder() :
   m_priority(false),
   m_elements(0),
   m_needsUpdate(true)
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
FWRPZDataProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   m_needsUpdate=true;
   if(0!=m_elements) {
      //m_elements->DestroyElements();
   }
}

void
FWRPZDataProxyBuilder::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   m_needsUpdate=false;
   if(0!=m_elements) {
      //m_elements->DestroyElements();
      m_elements->Destroy();
      //delete m_elements;
      m_elements=0;
   }
}

TEveElementList*
FWRPZDataProxyBuilder::getRhoPhiProduct() const
{
   if(m_needsUpdate) {
      m_needsUpdate=false;
      const_cast<FWRPZDataProxyBuilder*>(this)->build();
   }
   return m_elements;
}
TEveElementList*
FWRPZDataProxyBuilder::getRhoZProduct() const
{
   if(m_needsUpdate) {
      m_needsUpdate=false;
      const_cast<FWRPZDataProxyBuilder*>(this)->build();
   }
   return m_elements;
}

void
FWRPZDataProxyBuilder::build()
{
   if(0!= item()) {
      build(item(), &m_elements);

      setUserData(item(),m_elements,ids());
   }
}

void
FWRPZDataProxyBuilder::modelChangesImp(const FWModelIds& iIds)
{
   //I'm not sure I want to update the 3D version
   modelChanges(iIds,m_elements);
}

//
// const member functions
//

//
// static member functions
//

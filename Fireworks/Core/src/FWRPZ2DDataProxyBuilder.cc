// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZ2DDataProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FWRPZ2DDataProxyBuilder.cc,v 1.14 2008/10/28 14:17:14 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"

#include "Fireworks/Core/src/changeElementAndChildren.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
TEveCalo3D* FWRPZ2DDataProxyBuilder::m_caloRhoPhi = 0;
TEveCalo3D* FWRPZ2DDataProxyBuilder::m_caloRhoZ = 0;

//
// constructors and destructor
//
FWRPZ2DDataProxyBuilder::FWRPZ2DDataProxyBuilder():
  m_priority(false),
  m_rhoPhiElements(0),
  m_rhoZElements(0),
  m_rhoPhiNeedsUpdate(true),
  m_rhoZNeedsUpdate(true)
{
}

// FWRPZ2DDataProxyBuilder::FWRPZ2DDataProxyBuilder(const FWRPZ2DDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRPZ2DDataProxyBuilder::~FWRPZ2DDataProxyBuilder()
{
}

//
// assignment operators
//
// const FWRPZ2DDataProxyBuilder& FWRPZ2DDataProxyBuilder::operator=(const FWRPZ2DDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWRPZ2DDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWRPZ2DDataProxyBuilder::itemChangedImp(const FWEventItem*)
{
   /* inheriting classes own the elements
   if(0!=m_rhoPhiElements) {
      m_rhoPhiElements->DestroyElements();
   }
   if(0!=m_rhoZElements) {
      m_rhoZElements->DestroyElements();
   }
    */
   m_rhoPhiNeedsUpdate=true;
   m_rhoZNeedsUpdate=true;
}

void
FWRPZ2DDataProxyBuilder::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   if(0!=m_rhoPhiElements) {
      //m_rhoPhiElements->DestroyElements();
       m_rhoPhiElements->Destroy();
   }
   bool unique = m_rhoPhiElements!=m_rhoZElements;
   m_rhoPhiElements=0;
   if(unique) {
      if(0!=m_rhoZElements) {
         //m_rhoZElements->DestroyElements();
      }
      m_rhoZElements->Destroy();
      m_rhoZElements=0;
   }
}

TEveElementList*
FWRPZ2DDataProxyBuilder::getRhoPhiProduct() const
{
   if(m_rhoPhiNeedsUpdate) {
      m_rhoPhiNeedsUpdate=false;
      const_cast<FWRPZ2DDataProxyBuilder*>(this)->buildRhoPhi(&m_rhoPhiElements);
   }
   return m_rhoPhiElements;
}

TEveElementList*
FWRPZ2DDataProxyBuilder::getRhoZProduct() const
{
   if(m_rhoZNeedsUpdate) {
      m_rhoZNeedsUpdate=false;
      const_cast<FWRPZ2DDataProxyBuilder*>(this)->buildRhoZ(&m_rhoZElements);
   }
   return m_rhoZElements;
}



void
FWRPZ2DDataProxyBuilder::buildRhoPhi(TEveElementList** iObject)
{
  if(0!= item()) {
     buildRhoPhi(item(), iObject);
     setUserData(item(),*iObject,ids());
  }
}

void
FWRPZ2DDataProxyBuilder::buildRhoZ(TEveElementList** iObject)
{
  if(0!= item()) {
     buildRhoZ(item(), iObject);
     setUserData(item(),*iObject,ids());
  }
}

void
FWRPZ2DDataProxyBuilder::modelChangesImp(const FWModelIds& iIds)
{
}

//
// const member functions
//

//
// static member functions
//

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
// $Id: FWRPZ2DDataProxyBuilder.cc,v 1.10 2008/06/21 21:47:12 chrjones Exp $
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
  m_item(0),
  m_rhoPhiElements(0),
  m_rhoPhiZElements(0)
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
FWRPZ2DDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
   if(0 != m_item) {
      m_item->changed_.connect(boost::bind(&FWRPZ2DDataProxyBuilder::modelChangesRhoPhi,this,_1));
      m_item->changed_.connect(boost::bind(&FWRPZ2DDataProxyBuilder::modelChangesRhoZ,this,_1));
      m_item->goingToBeDestroyed_.connect(boost::bind(&FWRPZ2DDataProxyBuilder::itemBeingDestroyed,this,_1));
   }
}

void 
FWRPZ2DDataProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   delete m_rhoPhiElements;
   bool unique = m_rhoPhiElements!=m_rhoPhiZElements;
   m_rhoPhiElements=0;
   if(unique) {
      delete m_rhoPhiZElements;
      m_rhoPhiZElements=0;
   }
   for(std::vector<TEveElement*>::iterator it = m_rhoPhiProjs.begin(), itEnd = m_rhoPhiProjs.end();
       it != itEnd;
       ++it) {
      delete *it;
   }
   for(std::vector<TEveElement*>::iterator it = m_rhoZProjs.begin(), itEnd = m_rhoZProjs.end();
       it != itEnd;
       ++it) {
      delete *it;
   }
   m_rhoPhiProjs.clear();
   m_rhoZProjs.clear();
   m_ids.clear();
}


static void
setUserDataElementAndChildren(TEveElement* iElement, 
                              void* iInfo)
{
   iElement->SetUserData(iInfo);
   for(TEveElement::List_i itElement = iElement->BeginChildren(),
       itEnd = iElement->EndChildren();
       itElement != itEnd;
       ++itElement) {
      setUserDataElementAndChildren(*itElement, iInfo);
   }
}

static
void
setUserData(const FWEventItem* iItem,TEveElementList* iElements, std::vector<FWModelId>& iIds) {
   if(iElements &&  static_cast<int>(iItem->size()) == iElements->NumChildren() ) {
      int index=0;
      int largestIndex = iIds.size();
      if(iIds.size()<iItem->size()) {
         iIds.resize(iItem->size());
      }
      std::vector<FWModelId>::iterator itId = iIds.begin();
      for(TEveElement::List_i it = iElements->BeginChildren(), itEnd = iElements->EndChildren();
          it != itEnd;
          ++it,++itId,++index) {
         if(largestIndex<=index) {
            *itId=FWModelId(iItem,index);
         }
         setUserDataElementAndChildren(*it,&(*itId));
      }
   }
}



void
FWRPZ2DDataProxyBuilder::buildRhoPhi(TEveElementList** iObject)
{
  if(0!= m_item) {
     buildRhoPhi(m_item, iObject);
     setUserData(m_item,*iObject,m_ids);
  }
}

void
FWRPZ2DDataProxyBuilder::buildRhoZ(TEveElementList** iObject)
{
  if(0!= m_item) {
     buildRhoZ(m_item, iObject);
     setUserData(m_item,*iObject,m_ids);
  }
}

void 
FWRPZ2DDataProxyBuilder::modelChangesRhoPhi(const FWModelIds& iIds)
{
   std::for_each(m_rhoPhiProjs.begin(),
                 m_rhoPhiProjs.end(),
                 boost::bind(&FWRPZ2DDataProxyBuilder::modelChangesRhoPhi,
                             this,
                             iIds,
                             _1));
}

void 
FWRPZ2DDataProxyBuilder::modelChangesRhoZ(const FWModelIds& iIds)
{
   std::for_each(m_rhoZProjs.begin(),
                 m_rhoZProjs.end(),
                 boost::bind(&FWRPZ2DDataProxyBuilder::modelChangesRhoZ,
                             this,
                             iIds,
                             _1));
}

static 
void 
modelChanges(const FWEventItem* iItem, 
             const FWModelIds& iIds,
             TEveElement* iElements )
{
   if(0==iElements) {return;}
   //std::cout <<"modelChanged "<<iItem->size()<<" "<<iElements->GetNChildren()<<std::endl;
   assert(iItem && "item is not set");
   if ( static_cast<int>(iItem->size()) != iElements->NumChildren() ) {
      std::cout << "Inconsistent number of entries in the primary data collection and the proxy builder.\n" <<
	"Item name: " << iItem->name() << "\nN(data): " << iItem->size() <<
	"\nN(proxy): " << iElements->NumChildren() << std::endl;
   }
   assert(iItem && static_cast<int>(iItem->size()) == iElements->NumChildren() && "can not use default modelChanges implementation");
   TEveElement::List_i itElement = iElements->BeginChildren();
   int index = 0;
   for(FWModelIds::const_iterator it = iIds.begin(), itEnd = iIds.end();
       it != itEnd;
       ++it,++itElement,++index) {
      assert(itElement != iElements->EndChildren());         
      while(index < it->index()) {
         ++itElement;
         ++index;
         assert(itElement != iElements->EndChildren());         
      }
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(index);
      changeElementAndChildren(*itElement, info);
      (*itElement)->SetRnrSelf(info.displayProperties().isVisible());
      (*itElement)->SetRnrChildren(info.displayProperties().isVisible());
      (*itElement)->ElementChanged();
   }
}

void 
FWRPZ2DDataProxyBuilder::modelChangesRhoPhi(const FWModelIds& iIds, TEveElement* iElements)
{
   modelChanges(m_item,iIds,iElements);
}
void 
FWRPZ2DDataProxyBuilder::modelChangesRhoZ(const FWModelIds& iIds, TEveElement* iElements)
{
   modelChanges(m_item,iIds,iElements);
}

void 
FWRPZ2DDataProxyBuilder::addRhoPhiProj(TEveElement* iElement)
{
   
   //std::cout <<"setRhoPhiProj "<<m_item->name()<<" "<<iElement->GetRnrElName()<<" "<<iElement->GetNChildren()<<" "<<m_item->size()<<std::endl;
   assert(0!=iElement);
   m_rhoPhiProjs.push_back(iElement);
   //assert(0==iElement || iElement->GetNChildren() == m_item->size());
}
void 
FWRPZ2DDataProxyBuilder::addRhoZProj(TEveElement* iElement)
{
   assert(0!=iElement);
   m_rhoZProjs.push_back(iElement);
   //assert(0==iElement || iElement->GetNChildren() == m_item->size());
}

void 
FWRPZ2DDataProxyBuilder::clearRhoPhiProjs()
{
   m_rhoPhiProjs.clear();
}
void 
FWRPZ2DDataProxyBuilder::clearRhoZProjs()
{
   m_rhoZProjs.clear();
}

//
// const member functions
//

//
// static member functions
//

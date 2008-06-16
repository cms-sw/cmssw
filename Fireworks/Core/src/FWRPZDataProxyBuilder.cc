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
// $Id: FWRPZDataProxyBuilder.cc,v 1.11 2008/06/12 15:07:45 chrjones Exp $
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
FWRPZDataProxyBuilder::FWRPZDataProxyBuilder():
  m_item(0),
  m_rhoPhiProjs(),
  m_rhoZProjs()
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
  if(0 != m_item) {
     m_item->changed_.connect(boost::bind(&FWRPZDataProxyBuilder::modelChanges,this,_1));
     m_item->goingToBeDestroyed_.connect(boost::bind(&FWRPZDataProxyBuilder::itemBeingDestroyed,this,_1));

  }
}

void 
FWRPZDataProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   delete m_elements;
   m_elements=0;
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

void
FWRPZDataProxyBuilder::build(TEveElementList** iObject)
{
  if(0!= m_item) {
    build(m_item, iObject);
    m_elements=*iObject;

     if(m_elements &&  static_cast<int>(m_item->size()) == m_elements->NumChildren() ) {
        int index=0;
        int largestIndex = m_ids.size();
        if(m_ids.size()<m_item->size()) {
           m_ids.resize(m_item->size());
        }
        std::vector<FWModelId>::iterator itId = m_ids.begin();
        for(TEveElement::List_i it = m_elements->BeginChildren(), itEnd = m_elements->EndChildren();
            it != itEnd;
            ++it,++itId,++index) {
           if(largestIndex<=index) {
              *itId=FWModelId(m_item,index);
           }
           setUserDataElementAndChildren(*it,&(*itId));
        }
     }
  }
}

void 
FWRPZDataProxyBuilder::modelChanges(const FWModelIds& iIds)
{
   modelChanges(iIds,m_elements);
   std::for_each(m_rhoPhiProjs.begin(),
                 m_rhoPhiProjs.end(),
                 boost::bind(&FWRPZDataProxyBuilder::modelChanges,
                             this,
                             iIds,
                             _1));
   std::for_each(m_rhoZProjs.begin(),
                 m_rhoZProjs.end(),
                 boost::bind(&FWRPZDataProxyBuilder::modelChanges,
                             this,
                             iIds,
                             _1));
}


void 
FWRPZDataProxyBuilder::modelChanges(const FWModelIds& iIds,
                                    TEveElement* iElements )
{
   //std::cout <<"modelChanged "<<m_item->size()<<" "<<iElements->GetNChildren()<<std::endl;
   assert(m_item && static_cast<int>(m_item->size()) == iElements->NumChildren() && "can not use default modelChanges implementation");
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
FWRPZDataProxyBuilder::addRhoPhiProj(TEveElement* iElement)
{

   //std::cout <<"setRhoPhiProj "<<m_item->name()<<" "<<iElement->GetRnrElName()<<" "<<iElement->GetNChildren()<<" "<<m_item->size()<<std::endl;
   assert(0!=iElement);
   m_rhoPhiProjs.push_back(iElement);
   //assert(0==iElement || iElement->GetNChildren() == m_item->size());
}
void 
FWRPZDataProxyBuilder::addRhoZProj(TEveElement* iElement)
{
   assert(0!=iElement);
   m_rhoZProjs.push_back(iElement);
   //assert(0==iElement || iElement->GetNChildren() == m_item->size());
}

void 
FWRPZDataProxyBuilder::clearRhoPhiProjs()
{
   m_rhoPhiProjs.clear();
}
void 
FWRPZDataProxyBuilder::clearRhoZProjs()
{
   m_rhoZProjs.clear();
}


//
// const member functions
//

//
// static member functions
//

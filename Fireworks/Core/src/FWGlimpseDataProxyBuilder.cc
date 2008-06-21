// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseDataProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FWGlimpseDataProxyBuilder.cc,v 1.1 2008/06/19 06:57:27 dmytro Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"

#include "Fireworks/Core/src/changeElementAndChildren.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGlimpseDataProxyBuilder::FWGlimpseDataProxyBuilder():
  m_item(0),
  m_elements(0)
{
}

// FWGlimpseDataProxyBuilder::FWGlimpseDataProxyBuilder(const FWGlimpseDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWGlimpseDataProxyBuilder::~FWGlimpseDataProxyBuilder()
{
}

//
// assignment operators
//
// const FWGlimpseDataProxyBuilder& FWGlimpseDataProxyBuilder::operator=(const FWGlimpseDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWGlimpseDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWGlimpseDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
  if(0 != m_item) {
     m_item->changed_.connect(boost::bind(&FWGlimpseDataProxyBuilder::modelChanges,this,_1));
     m_item->goingToBeDestroyed_.connect(boost::bind(&FWGlimpseDataProxyBuilder::itemBeingDestroyed,this,_1));

  }
}

void 
FWGlimpseDataProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   delete m_elements;
   m_elements=0;
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
FWGlimpseDataProxyBuilder::build(TEveElementList** iObject)
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
FWGlimpseDataProxyBuilder::modelChanges(const FWModelIds& iIds)
{
   if(0!=m_elements) {
      modelChanges(iIds,m_elements);
   }
}


void 
FWGlimpseDataProxyBuilder::modelChanges(const FWModelIds& iIds,
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

//
// const member functions
//

//
// static member functions
//

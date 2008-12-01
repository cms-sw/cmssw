// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DDataProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FW3DDataProxyBuilder.cc,v 1.7 2008/11/14 15:32:32 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
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
FW3DDataProxyBuilder::FW3DDataProxyBuilder():
  m_item(0), m_elementHolder(new TEveElementList), m_modelsChanged(false), m_haveViews(false), m_mustBuild(true)
{
}

// FW3DDataProxyBuilder::FW3DDataProxyBuilder(const FW3DDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FW3DDataProxyBuilder::~FW3DDataProxyBuilder()
{
}

//
// assignment operators
//
// const FW3DDataProxyBuilder& FW3DDataProxyBuilder::operator=(const FW3DDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FW3DDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FW3DDataProxyBuilder::setHaveAWindow(bool iFlag)
{
   bool oldValue = m_haveViews;

   m_haveViews=iFlag;

   if(iFlag && !oldValue) {
      //this is our first view so may need to rerun our building
      if(m_mustBuild) {
         build();
      }
      if(m_modelsChanged) {
         applyChangesToAllModels();
      }
   }
}

TEveElement*
FW3DDataProxyBuilder::usedInScene()
{
   return m_elementHolder.get();
}


void
FW3DDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
  if(0 != m_item) {
     m_item->changed_.connect(boost::bind(&FW3DDataProxyBuilder::modelChanges,this,_1));
     m_item->goingToBeDestroyed_.connect(boost::bind(&FW3DDataProxyBuilder::itemBeingDestroyed,this,_1));
     m_item->itemChanged_.connect(boost::bind(&FW3DDataProxyBuilder::itemChanged,this,_1));
  }
}

void
FW3DDataProxyBuilder::itemChanged(const FWEventItem* iItem)
{
   if(m_haveViews) {
      build();
   } else {
      m_mustBuild=true;
   }
   m_modelsChanged=false;
}

void
FW3DDataProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   m_elementHolder->RemoveElements();
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
FW3DDataProxyBuilder::build()
{
  if(0!= m_item) {
     TEveElementList* newElements=0;
     bool notFirstTime = m_elementHolder->NumChildren();
     if(notFirstTime) {
        //we know the type since it is enforced in this routine so static_cast is safe
        newElements = static_cast<TEveElementList*>(*(m_elementHolder->BeginChildren()));
     }
     build(m_item, &newElements);
     if(!notFirstTime && newElements) {
        m_elementHolder->AddElement(newElements);
     }

     if(newElements &&  static_cast<int>(m_item->size()) == newElements->NumChildren() ) {
        int index=0;
        int largestIndex = m_ids.size();
        if(m_ids.size()<m_item->size()) {
           m_ids.resize(m_item->size());
        }
        std::vector<FWModelId>::iterator itId = m_ids.begin();
        for(TEveElement::List_i it = newElements->BeginChildren(),
            itEnd = newElements->EndChildren();
            it != itEnd;
            ++it,++itId,++index) {
           if(largestIndex<=index) {
              *itId=FWModelId(m_item,index);
           }
           setUserDataElementAndChildren(*it,&(*itId));
        }
     }
  }
   m_mustBuild=false;
}

void
FW3DDataProxyBuilder::modelChanges(const FWModelIds& iIds)
{
   if(m_haveViews) {
      if(0!=m_elementHolder->NumChildren()) {
         modelChanges(iIds,*(m_elementHolder->BeginChildren()));
      }
      m_modelsChanged=false;
   }else {
      m_modelsChanged=true;
   }
}

void
FW3DDataProxyBuilder::applyChangesToAllModels()
{
   applyChangesToAllModels(m_elementHolder->FirstChild());
   m_modelsChanged=false;
}

void
FW3DDataProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   FWModelIds ids(m_ids.begin(), m_ids.end());
   modelChanges(ids,iElements);
}

void
FW3DDataProxyBuilder::modelChanges(const FWModelIds& iIds,
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

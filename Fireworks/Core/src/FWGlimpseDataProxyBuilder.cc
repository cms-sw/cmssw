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
// $Id: FWGlimpseDataProxyBuilder.cc,v 1.7 2008/11/14 15:32:32 chrjones Exp $
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
#include "Fireworks/Core/interface/setUserDataElementAndChildren.h"

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
  m_item(0), m_elementHolder(new TEveElementList), m_modelsChanged(false), m_haveViews(false), m_scaler(0), m_mustBuild(true)
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
FWGlimpseDataProxyBuilder::setHaveAWindow(bool iFlag)
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
FWGlimpseDataProxyBuilder::usedInScene()
{
   return m_elementHolder.get();
}


void
FWGlimpseDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
  if(0 != m_item) {
     m_item->changed_.connect(boost::bind(&FWGlimpseDataProxyBuilder::modelChanges,this,_1));
     m_item->goingToBeDestroyed_.connect(boost::bind(&FWGlimpseDataProxyBuilder::itemBeingDestroyed,this,_1));
     m_item->itemChanged_.connect(boost::bind(&FWGlimpseDataProxyBuilder::itemChanged,this,_1));
  }
}

void
FWGlimpseDataProxyBuilder::itemChanged(const FWEventItem* iItem)
{
   itemChangedImp(iItem);
   if(m_haveViews) {
      build();
   } else {
      m_mustBuild=true;
   }
   m_modelsChanged=false;
}

void 
FWGlimpseDataProxyBuilder::itemChangedImp(const FWEventItem*)
{
}


void
FWGlimpseDataProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   m_elementHolder->RemoveElements();
   m_ids.clear();
}

void
FWGlimpseDataProxyBuilder::build()
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
           fireworks::setUserDataElementAndChildren(*it,&(*itId));
        }
     }
  }
   m_mustBuild=false;
}

void
FWGlimpseDataProxyBuilder::modelChanges(const FWModelIds& iIds)
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
FWGlimpseDataProxyBuilder::applyChangesToAllModels()
{
   applyChangesToAllModels(m_elementHolder->FirstChild());
   m_modelsChanged=false;
}

void
FWGlimpseDataProxyBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   FWModelIds ids(m_ids.begin(), m_ids.end());
   modelChanges(ids,iElements);
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
std::string 
FWGlimpseDataProxyBuilder::typeOfBuilder()
{
   return std::string();
}

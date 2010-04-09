// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones, Alja Mrak-Tadel
//         Created:  Thu Mar 18 14:12:00 CET 2010
// $Id: FWProxyBuilderBase.cc,v 1.1 2010/04/06 20:00:36 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
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
FWProxyBuilderBase::FWProxyBuilderBase():
   m_item(0), m_elementHolder(0), m_modelsChanged(false), m_haveViews(false), m_mustBuild(true)
{
   m_elementHolder = new TEveElementList();
   m_elementHolder->SetElementName("BuilderBaseHolder");
   m_elementHolder->IncDenyDestroy(); // el. should stay even not added to any scene
}

FWProxyBuilderBase::~FWProxyBuilderBase()
{
   m_elementHolder->DecDenyDestroy(); // auto destruct
}

//
// member functions
//

void
FWProxyBuilderBase::setItem(const FWEventItem* iItem)
{ 
   m_item = iItem;
   if(0 != m_item) {
      m_item->changed_.connect(boost::bind(&FWProxyBuilderBase::modelChanges,this,_1));
      m_item->goingToBeDestroyed_.connect(boost::bind(&FWProxyBuilderBase::itemBeingDestroyed,this,_1));
      m_item->itemChanged_.connect(boost::bind(&FWProxyBuilderBase::itemChanged,this,_1));
   }
}

void
FWProxyBuilderBase::setHaveAWindow(bool iFlag)
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

void
FWProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   m_elementHolder->RemoveElements();
   m_ids.clear();
}

void 
FWProxyBuilderBase::build()
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
         m_elementHolder->ProjectChild(newElements);
      }

      if(static_cast<int>(m_item->size()) == newElements->NumChildren() ) {
         int index=0;
         int largestIndex = m_ids.size();
         if(m_ids.size()<m_item->size()) {
            m_ids.resize(m_item->size());
         }
         std::vector<FWModelIdFromEveSelector>::iterator itId = m_ids.begin();
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
FWProxyBuilderBase::applyChangesToAllModels()
{
   applyChangesToAllModels(m_elementHolder->FirstChild());
   m_modelsChanged=false;
}

void
FWProxyBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   FWModelIds ids(m_ids.begin(), m_ids.end());
   modelChanges(ids,iElements);
}

//______________________________________________________________________________

void
FWProxyBuilderBase::modelChanges(const FWModelIds& iIds,
                                 TEveElement* iElements )
{
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
      if(specialModelChangeHandling(*it,*itElement)) {
         fireworks::setUserDataElementAndChildren(*itElement,&(ids()[it->index()]));
      }
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(index);
      changeElementAndChildren(*itElement, info);
      (*itElement)->SetRnrSelf(info.displayProperties().isVisible());
      (*itElement)->SetRnrChildren(info.displayProperties().isVisible());

      (*itElement)->ElementChanged();
   }
}

void
FWProxyBuilderBase::modelChanges(const FWModelIds& iIds)
{
   if(m_haveViews) {
      if(0!=m_elementHolder->NumChildren()) {
         modelChanges(iIds,*(m_elementHolder->BeginChildren()));
      }
      m_modelsChanged=false;
   } else {
      m_modelsChanged=true;
   }
}
//______________________________________________________________________________

void
FWProxyBuilderBase::itemChanged(const FWEventItem* iItem)
{ 
   itemChangedImp(iItem);
   if(m_haveViews) {
      build();
   } else {
      m_mustBuild=true;
   }
   m_modelsChanged=false;
}

void FWProxyBuilderBase::itemChangedImp(const FWEventItem*)
{
}

//______________________________________________________________________________

bool
FWProxyBuilderBase::canHandle(const FWEventItem& item)
{
   if (m_item)
      return (item.purpose() == m_item->purpose());

   return false;
}

//______________________________________________________________________________

void
FWProxyBuilderBase::attachToScene(const FWViewType& /*viewType*/, const std::string& /*purpose*/, TEveElementList* sceneHolder)
{
   // in trivial case ignore differences between view types
   // in trivial case proxy covers only one purpose
   sceneHolder->AddElement(m_elementHolder);
}

void
FWProxyBuilderBase::releaseFromSceneGraph(const FWViewType&)
{
   // to do ...
}

//______________________________________________________________________________

bool
FWProxyBuilderBase::willHandleInteraction()
{
   // by default view manager handles interaction
   return false;
}

void
FWProxyBuilderBase::setInteractionList(std::vector<TEveCompound* > const * interactionList, std::string& purpose )
{
}


bool
FWProxyBuilderBase::specialModelChangeHandling(const FWModelId&, TEveElement*)
{
   return false;
}

TEveElementList*
FWProxyBuilderBase::getProduct()
{
  return m_elementHolder;
}

//
// const member functions
//

const fireworks::Context&
FWProxyBuilderBase::context() const
{
   return m_item->context();
}

int
FWProxyBuilderBase::layer() const
{
   return m_item->layer();
}

//
// static member functions
//

std::string FWProxyBuilderBase::typeOfBuilder()
{
   return std::string();
}

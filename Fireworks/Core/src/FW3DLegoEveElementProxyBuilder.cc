// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoEveElementProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul  5 11:13:22 EDT 2008
// $Id: FW3DLegoEveElementProxyBuilder.cc,v 1.7 2009/10/28 14:46:16 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"
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
FW3DLegoEveElementProxyBuilder::FW3DLegoEveElementProxyBuilder()
{
}


FW3DLegoEveElementProxyBuilder::~FW3DLegoEveElementProxyBuilder()
{
}


//
// member functions
//

void
FW3DLegoEveElementProxyBuilder::attach(TEveElement* iElement,
                                       TEveCaloDataHist* iHist)
{
   m_elementHolder = new TEveElementList("FW3DLegoEveElementProxyBuilder Holder");
   iElement->AddElement(m_elementHolder);
}

void
FW3DLegoEveElementProxyBuilder::modelChangesImp(const FWModelIds& iIds)
{
   modelChanges(iIds,static_cast<TEveElementList*>(m_elementHolder->FirstChild()));
}

void
FW3DLegoEveElementProxyBuilder::itemChangedImp(const FWEventItem*)
{
}

void
FW3DLegoEveElementProxyBuilder::applyChangesToAllModels()
{
   FWModelIds allIds(ids().begin(),ids().end());
   modelChangesImp(allIds);
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
setUserData(const FWEventItem* iItem,TEveElementList* iElements, std::vector<FWModelIdFromEveSelector>& iIds) {
   if(iElements &&  static_cast<int>(iItem->size()) == iElements->NumChildren() ) {
      int index=0;
      int largestIndex = iIds.size();
      if(iIds.size()<iItem->size()) {
         iIds.resize(iItem->size());
      }
      std::vector<FWModelIdFromEveSelector>::iterator itId = iIds.begin();
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
FW3DLegoEveElementProxyBuilder::build()
{
   if(0==item()) { return;}
   TEveElementList* newElements=0;
   bool notFirstTime = m_elementHolder->NumChildren();
   if(notFirstTime) {
      //we know the type since it is enforced in this routine so static_cast is safe
      newElements = static_cast<TEveElementList*>(*(m_elementHolder->BeginChildren()));
   }
   build(item(), &newElements);
   if(!notFirstTime && newElements) {
      m_elementHolder->AddElement(newElements);
   }
   setUserData(item(),newElements,ids());
}

void
FW3DLegoEveElementProxyBuilder::modelChanges(const FWModelIds& iIds,
                                             TEveElement* iElements )
{
   //std::cout <<"modelChanged "<<m_item->size()<<" "<<iElements->GetNChildren()<<std::endl;
   if (!iElements) return;
   assert(item() && static_cast<int>(item()->size()) == iElements->NumChildren() && "can not use default modelChanges implementation");
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
         setUserDataElementAndChildren(*itElement,&(ids()[it->index()]));
      }
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(index);
      changeElementAndChildren(*itElement, info);
      (*itElement)->SetRnrSelf(info.displayProperties().isVisible());
      (*itElement)->SetRnrChildren(info.displayProperties().isVisible());
      (*itElement)->ElementChanged();
   }
}

bool 
FW3DLegoEveElementProxyBuilder::specialModelChangeHandling(const FWModelId&, TEveElement*) { return false;}

void
FW3DLegoEveElementProxyBuilder::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   m_elementHolder->DestroyElements();
}

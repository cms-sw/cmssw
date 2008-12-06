// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilderBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jun 28 09:51:35 PDT 2008
// $Id: FWRPZDataProxyBuilderBase.cc,v 1.7 2008/11/26 01:55:31 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRhoPhiZView.h"
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
FWRPZDataProxyBuilderBase::FWRPZDataProxyBuilderBase():
m_item(0),
m_rpviews(0),
m_rzviews(0),
m_rhoPhiProjs(),
m_rhoZProjs(),
m_ids(),
m_layer(0),
m_modelsChanged(true)
{
}

// FWRPZDataProxyBuilderBase::FWRPZDataProxyBuilderBase(const FWRPZDataProxyBuilderBase& rhs)
// {
//    // do actual copying here;
// }

FWRPZDataProxyBuilderBase::~FWRPZDataProxyBuilderBase()
{
   //We want all views to get rid of these so we destroy the contents
   m_rhoPhiProjs.DestroyElements();
   m_rhoZProjs.DestroyElements();
}

//
// assignment operators
//
// const FWRPZDataProxyBuilderBase& FWRPZDataProxyBuilderBase::operator=(const FWRPZDataProxyBuilderBase& rhs)
// {
//   //An exception safe implementation is
//   FWRPZDataProxyBuilderBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWRPZDataProxyBuilderBase::setItem(const FWEventItem* iItem)
{
   m_item = iItem;
   if(0 != m_item) {
      m_item->itemChanged_.connect(boost::bind(&FWRPZDataProxyBuilderBase::itemChanged,this,_1));
      m_item->changed_.connect(boost::bind(&FWRPZDataProxyBuilderBase::modelChanges,this,_1));
      m_item->goingToBeDestroyed_.connect(boost::bind(&FWRPZDataProxyBuilderBase::itemBeingDestroyed,this,_1));
      m_layer = m_item->layer();
   }
}

void
FWRPZDataProxyBuilderBase::itemChanged(const FWEventItem* iItem)
{
   //We want all views to get rid of these so we destroy the contents
   m_rhoPhiProjs.DestroyElements();
   m_rhoZProjs.DestroyElements();

   m_layer = iItem->layer();
   itemChangedImp(iItem);

   std::for_each(m_rpviews->begin(),m_rpviews->end(),
                 boost::bind(&FWRPZDataProxyBuilderBase::attachToRhoPhiView,
                             this,_1));
   std::for_each(m_rzviews->begin(),m_rzviews->end(),
                 boost::bind(&FWRPZDataProxyBuilderBase::attachToRhoZView,
                             this,_1));
   m_modelsChanged=false;
}

void
FWRPZDataProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;
   //We want all views to get rid of these so we destroy the contents
   m_rhoPhiProjs.DestroyElements();
   m_rhoZProjs.DestroyElements();
   m_ids.clear();
   itemBeingDestroyedImp(iItem);
}



void
FWRPZDataProxyBuilderBase::modelChanges(const FWModelIds& iIds,
             TEveElement* iElements )
{
   if(0==iElements) {return;}
   //std::cout <<"modelChanged "<<m_item->size()<<" "<<iElements->GetNChildren()<<std::endl;
   assert(m_item && "item is not set");
   if ( static_cast<int>(m_item->size()) != iElements->NumChildren() ) {
      std::cout << "Inconsistent number of entries in the primary data collection and the proxy builder.\n" <<
      "Item name: " << m_item->name() << "\nN(data): " << m_item->size() <<
      "\nN(proxy): " << iElements->NumChildren() << std::endl;
   }
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
FWRPZDataProxyBuilderBase::modelChanges(const FWModelIds& iIds)
{
   if(!m_rpviews->empty() || !m_rzviews->empty()) {
      std::for_each(m_rhoPhiProjs.BeginChildren(),
                    m_rhoPhiProjs.EndChildren(),
                    boost::bind(&FWRPZDataProxyBuilderBase::modelChanges,
                                this,
                                iIds,
                                _1));
      std::for_each(m_rhoZProjs.BeginChildren(),
                    m_rhoZProjs.EndChildren(),
                    boost::bind(&FWRPZDataProxyBuilderBase::modelChanges,
                                this,
                                iIds,
                                _1));


      modelChangesImp(iIds);
      m_modelsChanged=false;
   } else {
      m_modelsChanged=true;
   }
}

void
FWRPZDataProxyBuilderBase::setViews(
              std::vector<boost::shared_ptr<FWRhoPhiZView> >* iRhoPhoViews,
              std::vector<boost::shared_ptr<FWRhoPhiZView> >* iRhoZViews)
{
   m_rpviews=iRhoPhoViews;
   m_rzviews=iRhoZViews;
   assert(0!=m_rpviews);
   assert(0!=m_rzviews);

   std::for_each(m_rpviews->begin(),m_rpviews->end(),
                 boost::bind(&FWRPZDataProxyBuilderBase::attachToRhoPhiView,
                             this,_1));
   std::for_each(m_rzviews->begin(),m_rzviews->end(),
                 boost::bind(&FWRPZDataProxyBuilderBase::attachToRhoZView,
                             this,_1));
}

void
FWRPZDataProxyBuilderBase::attachToRhoPhiView(boost::shared_ptr<FWRhoPhiZView> iView)
{
   //std::cout <<"attachToRhoPhiView " << typeid(*this).name() <<std::endl;
   if(m_item && m_item->hasEvent()) {
      if(TEveElementList* prod = getRhoPhiProduct()) {
         addRhoPhiProj( iView->importElements(prod,layer()));
      }
      if(m_modelsChanged) {
         if(m_rhoPhiProjs.HasChildren()) {
            //it appears that passing iterators to 0 length vector to a set causes a crash
            applyChangesToAllModels(*(m_rhoPhiProjs.BeginChildren()));
         }
         m_modelsChanged=false;
      }
   }
}

void
FWRPZDataProxyBuilderBase::attachToRhoZView(boost::shared_ptr<FWRhoPhiZView> iView)
{
   //std::cout <<"attachToRhoZView " << typeid(*this).name() <<std::endl;

   if(m_item && m_item->hasEvent()) {
      if(TEveElementList* prod = getRhoZProduct()) {
         addRhoZProj( iView->importElements(prod,layer()));
      }
      if(m_modelsChanged) {
         if(m_rhoZProjs.HasChildren()) {
            applyChangesToAllModels(*(m_rhoZProjs.BeginChildren()));
         }
         m_modelsChanged=false;
      }
   }
}

void
FWRPZDataProxyBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   //it appears that passing iterators to 0 length vector to a set causes a crash
   if(m_item->size()) {
      //cassert(m_ids.size() >= m_item->size());
      if ( !( m_ids.size() >= m_item->size() ) ) return;
      //any of them may have changed so we need to update them all
      FWModelIds ids(m_ids.begin(),m_ids.begin()+m_item->size());
      modelChanges(ids,iElements);
   }
}

void
FWRPZDataProxyBuilderBase::addRhoPhiProj(TEveElement* iElement)
{
   //std::cout <<"addRhoPhiProj "<<iElement->NumChildren()<<std::endl;
   m_rhoPhiProjs.AddElement(iElement);
}

void
FWRPZDataProxyBuilderBase::addRhoZProj(TEveElement* iElement)
{
   //std::cout <<"addRhoZProj "<<iElement->NumChildren()<<std::endl;
   m_rhoZProjs.AddElement(iElement);
}


//
// const member functions
//

//
// static member functions
//

static
void
setUserDataElementAndChildren(TEveElement* iElement,
                              const void* iInfo)
{
   iElement->SetUserData(const_cast<void*>(iInfo));
   for(TEveElement::List_i itElement = iElement->BeginChildren(),
       itEnd = iElement->EndChildren();
       itElement != itEnd;
       ++itElement) {
      setUserDataElementAndChildren(*itElement, iInfo);
   }
}

void
FWRPZDataProxyBuilderBase::setUserData(const FWEventItem* iItem,TEveElementList* iElements, std::vector<FWModelId>& iIds) {
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

std::string 
FWRPZDataProxyBuilderBase::typeOfBuilder()
{
   return std::string();
}

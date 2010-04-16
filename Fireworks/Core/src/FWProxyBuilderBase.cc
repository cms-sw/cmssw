// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones, Matevz Tadel, Alja Mrak-Tadel
//         Created:  Thu Mar 18 14:12:00 CET 2010
// $Id: FWProxyBuilderBase.cc,v 1.4 2010/04/16 14:41:23 amraktad Exp $
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
   m_item(0),
   m_modelsChanged(false),
   m_haveViews(false),
   m_mustBuild(true),
   m_layer(0)
{
}

FWProxyBuilderBase::~FWProxyBuilderBase()
{
   m_products.clear();
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
   for (Product_it i = m_products.begin(); i!= m_products.end(); i++)
   {
      delete (*i);
   }

   m_products.clear();
   m_ids.clear();
}

void 
FWProxyBuilderBase::build()
{
   if (m_item)
   {
      bool firstTime = m_products.size();
      clean();

      bool initIds = true;
      for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
      {
         TEveElementList* elms = (*i)->m_elements;

         if (hasSingleProduct())
         {
            build(m_item, elms);
         }
         else
         {
            buildViewType(m_item, elms, (*i)->m_viewType);
         }

         // Project all children of current product.
         // If product is not registered into any projection-manager,
         // this does nothing.
         // It might be cleaner to check view-type / supported view-types.
         if (firstTime) setProjectionLayer(item()->layer());
         elms->ProjectAllChildren();

         if (elms && static_cast<int>(m_item->size()) == elms->NumChildren())
         {
            if (initIds)
            {
               int largestIndex = m_ids.size();
               if (m_ids.size()<m_item->size())
               {
                  m_ids.resize(m_item->size());
               }

               std::vector<FWModelIdFromEveSelector>::iterator itId = m_ids.begin();
               for (int index = 0; index < elms->NumChildren(); ++itId,++index)
               {
                  if (index >= largestIndex)
                  {
                     *itId = FWModelId(m_item,index);
                  }
               }
               initIds = false;
            }
         
        
            std::vector<FWModelIdFromEveSelector>::iterator itId = m_ids.begin();
            for (TEveElement::List_i it = elms->BeginChildren(),
                    itEnd = elms->EndChildren();
                 it != itEnd;
                 ++it,++itId)
            {
               fireworks::setUserDataElementAndChildren(*it, &(*itId));
            }
         }
      }
   }
   m_mustBuild = false;
}

void
FWProxyBuilderBase::applyChangesToAllModels()
{
   for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
      applyChangesToAllModels((*i)->m_elements);

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
                                 TEveElement* iElements)
{
   //   printf("modelChange %d %d \n", iElements->NumChildren() ,  m_item->size()); fflush(stdout);
   assert(m_item && static_cast<int>(m_item->size()) == iElements->NumChildren() && "can not use default modelChanges implementation");

   TEveElement::List_i itElement = iElements->BeginChildren();
   int index = 0;
   for (FWModelIds::const_iterator it = iIds.begin(), itEnd = iIds.end();
	it != itEnd;
	++it,++itElement,++index)
   {
      assert(itElement != iElements->EndChildren());
      while (index < it->index())
      {
         ++itElement;
         ++index;
         assert(itElement != iElements->EndChildren());
      }
      if (specialModelChangeHandling(*it,*itElement))
      {
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
    for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
    {
      modelChanges(iIds, (*i)->m_elements);
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

   if (iItem->layer() != m_layer)
      setProjectionLayer(iItem->layer());

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
bool
FWProxyBuilderBase::hasPerViewProduct()
{
  // by default all scenes are shared.
  return false;
}

bool
FWProxyBuilderBase::hasSingleProduct()
{
  // trivial projections use only one product
  return true;
}

TEveElementList*
FWProxyBuilderBase::createProduct(const FWViewType::EType viewType, FWViewContext* viewContext)
{
   if (hasSingleProduct() == false || m_products.empty())
   {
      m_products.push_back(new Product(viewType, viewContext));

      if (viewContext)
      {
         // TODO viewContext.scaleChanged_connect(boost::bind(&FWProxyBuilderBase::setScale, this))
      }

      if (item()) 
      {
         // debug info for eve browser       
         m_products.back()->m_elements->SetElementName(item()->name().c_str());
      }
   }

   return m_products.back()->m_elements;
}

void
FWProxyBuilderBase::releaseFromSceneGraph(const FWViewType&)
{
  // TODO ...
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

void
FWProxyBuilderBase::clean()
{
   // Cleans local common element list.
   for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
   {
      if ((*i)->m_elements)
         (*i)->m_elements->DestroyElements();
   }

   cleanLocal();
}

void
FWProxyBuilderBase::cleanLocal()
{
   // Cleans local common element list.
}


void
FWProxyBuilderBase::build(const FWEventItem*, TEveElementList*)
{
   // TODO when design is complete this has to be abstract function
   assert(false && "Function used but not implemented.");
}

void 
FWProxyBuilderBase::buildViewType(const FWEventItem*, TEveElementList*, FWViewType::EType)
{
   // TODO when design is complete this has to be abstract function
   assert(false && "Function used but not implemented.");
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

void
FWProxyBuilderBase::setProjectionLayer(float layer)
{
   m_layer = layer;
   for (Product_it pIt = m_products.begin(); pIt != m_products.end(); ++pIt)
   {
      TEveProjectable* pable = static_cast<TEveProjectable*>((*pIt)->m_elements);
      for (TEveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
         (*i)->SetDepth(m_layer);
   }
}

//
// static member functions
//

std::string FWProxyBuilderBase::typeOfBuilder()
{
   return std::string();
}

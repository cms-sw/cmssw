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
// $Id: FWProxyBuilderBase.cc,v 1.8 2010/04/22 10:20:56 matevz Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWInteractionList.h"


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
   m_interactionList(0),
   m_item(0),
   m_modelsChanged(false),
   m_haveWindow(false),
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
}

void
FWProxyBuilderBase::setHaveWindow(bool iFlag)
{
   bool oldValue = m_haveWindow;
   m_haveWindow=iFlag;

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
}

void 
FWProxyBuilderBase::build()
{
   if (m_item)
   {
      bool firstTime = m_products.size();
      clean();
      for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
      {
         // printf("build() %s \n", m_item->name().c_str());
         TEveElementList* elms = (*i)->m_elements;

         if (haveSingleProduct())
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


         if (m_interactionList)
         {
            unsigned int idx = 0;
            for (TEveElement::List_i it = elms->BeginChildren(); it !=  elms->EndChildren(); ++it)
               m_interactionList->added(*it, idx++);
         }
      }
   }
   m_mustBuild = false;
}

void
FWProxyBuilderBase::applyChangesToAllModels()
{
   for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
      applyChangesToAllModels((*i)->m_elements, (*i)->m_viewType);

   m_modelsChanged=false;
}

void
FWProxyBuilderBase::applyChangesToAllModels(TEveElement* elms, FWViewType::EType viewType)
{
   FWModelIds ids(m_ids.begin(), m_ids.end());
   modelChanges(ids, elms, viewType);
}

//______________________________________________________________________________
void
FWProxyBuilderBase::modelChanges(const FWModelIds& iIds,
                                 TEveElement* elms, FWViewType::EType viewType)
{
   assert(m_item && static_cast<int>(m_item->size()) == elms->NumChildren() && "can not use default modelChanges implementation");

   TEveElement::List_i itElement = elms->BeginChildren();
   int index = 0;
   for (FWModelIds::const_iterator it = iIds.begin(), itEnd = iIds.end();
	it != itEnd;
	++it,++itElement,++index)
   {
      assert(itElement != elms->EndChildren());
      while (index < it->index())
      {
         ++itElement;
         ++index;
         assert(itElement != elms->EndChildren());
      }
      if (specialModelChangeHandling(*it,*itElement, viewType))
      {
         elms->ProjectChild(*itElement);
      }
   }
}

void
FWProxyBuilderBase::modelChanges(const FWModelIds& iIds)
{
  if(m_haveWindow) {
    for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
    {
       modelChanges(iIds, (*i)->m_elements, (*i)->m_viewType);
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
   if (iItem->layer() != m_layer)
      setProjectionLayer(iItem->layer());

   if(m_haveWindow) {
      build();
   } else {
      m_mustBuild=true;
   }
   m_modelsChanged=false;
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

TEveElementList*
FWProxyBuilderBase::createProduct(const FWViewType::EType viewType, FWViewContext* viewContext)
{
   if (haveSingleProduct() == false || m_products.empty())
   {
      m_products.push_back(new Product(viewType, viewContext));

      if (viewContext)
      {
         // TODO auto-scale  viewContext.scaleChanged_connect(boost::bind(&FWProxyBuilderBase::setScale, this))
      }

      if (item()) 
      {
         // debug info in eve browser       
         m_products.back()->m_elements->SetElementName(item()->name().c_str());
      }
   }

   return m_products.back()->m_elements;
}

//------------------------------------------------------------------------------

void
FWProxyBuilderBase::setInteractionList(FWInteractionList* l, const std::string& /*purpose*/ )
{
   // Called if willHandleInteraction() returns false. Purpose ignored by default.

   m_interactionList = l;
}


bool
FWProxyBuilderBase::specialModelChangeHandling(const FWModelId&, TEveElement*, FWViewType::EType)
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
   assert(false && "Function used, but not implemented.");
}

void 
FWProxyBuilderBase::buildViewType(const FWEventItem*, TEveElementList*, FWViewType::EType)
{
   // TODO when design is complete this has to be abstract function
   assert(false && "Function used, but not implemented.");
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

//------------------------------------------------------------------------------

void
FWProxyBuilderBase::setupAddElement(TEveElement* el, TEveElement* parent, bool color) const
{
   setupElement(el, color);
   parent->AddElement(el);
}

void
FWProxyBuilderBase::setupElement(TEveElement* el, bool color) const
{
   el->CSCTakeAnyParentAsMaster();
   el->SetPickable(true);

   if (color)
   {
      el->CSCApplyMainColorToMatchingChildren();
      el->SetMainColor(m_item->defaultDisplayProperties().color());
   }
}

//------------------------------------------------------------------------------

TEveCompound*
FWProxyBuilderBase::createCompound(bool set_color, bool propagate_color_to_all_children) const
{
   TEveCompound* c = new TEveCompound();
   c->CSCTakeAnyParentAsMaster();
   c->CSCImplySelectAllChildren();
   c->SetPickable(true);
   if (set_color)
   {
      c->SetMainColor(m_item->defaultDisplayProperties().color());
   }
   if (propagate_color_to_all_children)
      c->CSCApplyMainColorToAllChildren();
   else
      c->CSCApplyMainColorToMatchingChildren();
   return c;
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

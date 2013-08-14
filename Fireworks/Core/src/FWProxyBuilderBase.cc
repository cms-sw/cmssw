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
// $Id: FWProxyBuilderBase.cc,v 1.32 2010/10/29 10:42:02 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveProjectionManager.h"
#include "TEveSelection.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWInteractionList.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/fwLog.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//


FWProxyBuilderBase::Product::Product(FWViewType::EType t, const FWViewContext* c) : m_viewType(t), m_viewContext(c), m_elements(0)
{
   m_elements = new TEveElementList("ProxyProduct");
   m_elements->IncDenyDestroy();
}


FWProxyBuilderBase::Product::~Product()
{
   // remove product from projected scene (RhoPhi or RhoZ)
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(m_elements);
   // don't have to check cast, because TEveElementList is TEveProjectable
   for (TEveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
   {
      TEveElement* projected  = (*i)->GetProjectedAsElement();
      (*projected->BeginParents())->RemoveElement(projected);
   }

   // remove from 3D scenes
   while (m_elements->HasParents())
   {
      TEveElement* parent = *m_elements->BeginParents();
      parent->RemoveElement(m_elements);
   }

   m_elements->Annihilate();
}

//______________________________________________________________________________

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
   }
}

void
FWProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   m_item=0;

   cleanLocal();

   for (Product_it i = m_products.begin(); i!= m_products.end(); i++)
   {

      (*i)->m_scaleConnection.disconnect();
      delete (*i);
   }

   m_products.clear();
}

void 
FWProxyBuilderBase::build()
{
   if (m_item)
   {
      try 
      {
         size_t itemSize = m_item->size(); //cashed

         clean();
         for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
         {
            // printf("build() %s \n", m_item->name().c_str());
            TEveElementList* elms = (*i)->m_elements;
            size_t oldSize = elms->NumChildren();

            if (haveSingleProduct())
            {
               build(m_item, elms, (*i)->m_viewContext);
            }
            else
            {
               buildViewType(m_item, elms, (*i)->m_viewType, (*i)->m_viewContext);
            }

            // Project all children of current product.
            // If product is not registered into any projection-manager,
            // this does nothing.          
            TEveProjectable* pable = dynamic_cast<TEveProjectable*>(elms);
            if (pable->HasProjecteds())
            {
               for (TEveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
               {
                  TEveProjectionManager *pmgr = (*i)->GetManager();
                  Float_t oldDepth = pmgr->GetCurrentDepth();
                  pmgr->SetCurrentDepth(item()->layer());
                  size_t cnt = 0;

                  TEveElement* projectedAsElement = (*i)->GetProjectedAsElement();
                  TEveElement::List_i parentIt = projectedAsElement->BeginChildren();
                  for (TEveElement::List_i prodIt = elms->BeginChildren(); prodIt != elms->EndChildren(); ++prodIt, ++cnt)
                  {
                     if (cnt < oldSize)
                     {  
                        // reused projected holder
                        pmgr->SubImportChildren(*prodIt, *parentIt);
                        ++parentIt;
                     }
                     else if (cnt < itemSize) 
                     {
                        // new product holder
                        pmgr->SubImportElements(*prodIt, projectedAsElement);
                     }
                     else
                     {
                        break;
                     }
                  }
                  pmgr->SetCurrentDepth(oldDepth);
               }
            }
         

            if (m_interactionList && itemSize > oldSize)
            {
               TEveElement::List_i elIt = elms->BeginChildren();
               for (size_t cnt = 0; cnt < itemSize; ++cnt, ++elIt)
               {
                  if (cnt >= oldSize )
                     m_interactionList->added(*elIt, cnt);
               }
            }
         }
      }
      catch (const std::runtime_error& iException)
      { 
         fwLog(fwlog::kError) << "Caught exception in build function for item " << m_item->name() << ":\n"
                              << iException.what() << std::endl;
         exit(1);
      }
   }
   m_mustBuild = false;
}

//______________________________________________________________________________
void
FWProxyBuilderBase::modelChanges(const FWModelIds& iIds, Product* p)
{
   TEveElementList* elms = p->m_elements;
   assert(m_item && static_cast<int>(m_item->size()) <= elms->NumChildren() && "can not use default modelChanges implementation");

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
      if (visibilityModelChanges(*it, *itElement, p->m_viewType, p->m_viewContext))
      {
         elms->ProjectChild(*itElement);
      }
      else
      {
        localModelChanges(*it, *itElement, p->m_viewType, p->m_viewContext);
      }
   }
}

void
FWProxyBuilderBase::modelChanges(const FWModelIds& iIds)
{
  if(m_haveWindow) {
    for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
    {
       modelChanges(iIds, *i);
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
FWProxyBuilderBase::createProduct(const FWViewType::EType viewType, const FWViewContext* viewContext)
{
   if ( havePerViewProduct(viewType) == false && m_products.empty() == false)
   {
      if (haveSingleProduct())
      {
         return m_products.back()->m_elements;
      }
      else
      {
         for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
         {
            if (viewType == (*i)->m_viewType)  
               return (*i)->m_elements;
         }
      }
   }

   // printf("new product %s for item %s \n", FWViewType::idToName(viewType).c_str(), item()->name().c_str()); fflush(stdout);
 
   Product* product = new Product(viewType, viewContext);
   m_products.push_back(product);
   if (viewContext)
   {
      product->m_scaleConnection = viewContext->scaleChanged_.connect(boost::bind(&FWProxyBuilderBase::scaleChanged, this, _1));
   }

   if (item()) 
   {
      // debug info in eve browser       
      product->m_elements->SetElementName(item()->name().c_str());
   }
   return product->m_elements;
}

//______________________________________________________________________________

void
FWProxyBuilderBase::removePerViewProduct(FWViewType::EType type, const FWViewContext* vc)
{
   for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
   {  
      if (havePerViewProduct(type) &&  (*i)->m_viewContext == vc)
      { 
         if ((*i)->m_elements)
            (*i)->m_elements->DestroyElements();

         if ( (*i)->m_viewContext)
            (*i)->m_scaleConnection.disconnect();

         delete (*i);
         m_products.erase(i);
         break;
      }
   }
}

//------------------------------------------------------------------------------

void
FWProxyBuilderBase::setInteractionList(FWInteractionList* l, const std::string& /*purpose*/ )
{
   // Called if willHandleInteraction() returns false. Purpose ignored by default.

   m_interactionList = l;
}


bool
FWProxyBuilderBase::visibilityModelChanges(const FWModelId&, TEveElement*, FWViewType::EType, const FWViewContext*)
{
   return false;
}

void
FWProxyBuilderBase::localModelChanges(const FWModelId&, TEveElement*, FWViewType::EType, const FWViewContext*)
{
   // Nothing to be done in base class.
   // Visibility, main color and main transparency are handled through FWInteractionList.
}


void
FWProxyBuilderBase::scaleChanged(const FWViewContext* vc)
{
   for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
   {  
      if ( havePerViewProduct((*i)->m_viewType) && (*i)->m_viewContext == vc)
      {
         scaleProduct((*i)->m_elements, (*i)->m_viewType, (*i)->m_viewContext);
      }
   }
   gEve->Redraw3D();
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
FWProxyBuilderBase::build(const FWEventItem*, TEveElementList*, const FWViewContext*)
{
   assert("virtual build(const FWEventItem*, TEveElementList*, const FWViewContext*) not implemented by inherited class");
}

void 
FWProxyBuilderBase::buildViewType(const FWEventItem*, TEveElementList*, FWViewType::EType, const FWViewContext*)
{
   assert("virtual buildViewType(const FWEventItem*, TEveElementList*, FWViewType::EType, const FWViewContext*) not implemented by inherited class");
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

/** This method is invoked to setup the per element properties of the various
    objects being drawn.
  */
void
FWProxyBuilderBase::setupElement(TEveElement* el, bool color) const
{
   el->CSCTakeAnyParentAsMaster();
   el->SetPickable(true);

   if (color)
   {
      el->CSCApplyMainColorToMatchingChildren();
      el->CSCApplyMainTransparencyToMatchingChildren();
      el->SetMainColor(m_item->defaultDisplayProperties().color());
      assert((m_item->defaultDisplayProperties().transparency() >= 0)
             && (m_item->defaultDisplayProperties().transparency() <= 100));
      el->SetMainTransparency(m_item->defaultDisplayProperties().transparency());
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
      c->SetMainTransparency(m_item->defaultDisplayProperties().transparency());
   }
   if (propagate_color_to_all_children)
   {
      c->CSCApplyMainColorToAllChildren();
      c->CSCApplyMainTransparencyToAllChildren();
   }
   else
   {
      c->CSCApplyMainColorToMatchingChildren();
      c->CSCApplyMainTransparencyToMatchingChildren();
   }
   return c;
}

void
FWProxyBuilderBase::increaseComponentTransparency(unsigned int index, TEveElement* holder,
                                                    const std::string& name, Char_t transpOffset)
{
   // Helper function to increse transparency of certain components.

   const FWDisplayProperties& dp = item()->modelInfo(index).displayProperties();
   Char_t transp = TMath::Min(100, transpOffset + (100 - transpOffset) * dp.transparency() / 100);
   TEveElement::List_t matches;
   holder->FindChildren(matches, name.c_str());
   for (TEveElement::List_i m = matches.begin(); m != matches.end(); ++m)
   {
      (*m)->SetMainTransparency(transp);
   }
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

bool FWProxyBuilderBase::representsSubPart()
{
   return false;
}

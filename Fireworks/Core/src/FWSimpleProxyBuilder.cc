// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseSimpleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones, Alja Mrak-Tadel
//         Created:  Tue March 28 09:46:41 EST 2010
// $Id: FWSimpleProxyBuilder.cc,v 1.13 2010/10/22 14:34:45 amraktad Exp $
//

// system include files
#include <memory>

// user include files
#include "TEveCompound.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWSimpleProxyBuilder::FWSimpleProxyBuilder(const std::type_info& iType) :
   m_helper(iType)
{
}

// FWSimpleProxyBuilder::FWSimpleProxyBuilder(const FWSimpleProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWSimpleProxyBuilder::~FWSimpleProxyBuilder()
{
}

//
// assignment operators
//
// const FWSimpleProxyBuilder& FWSimpleProxyBuilder::operator=(const FWSimpleProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWSimpleProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWSimpleProxyBuilder::clean()
{
   for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
   {
      if ((*i)->m_elements)
      {
         TEveElement* elms = (*i)->m_elements;
         for (TEveElement::List_i it = elms->BeginChildren(); it != elms->EndChildren(); ++it)
            (*it)->DestroyElements();
      }
   }

   cleanLocal();
}

void
FWSimpleProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   if (iItem)
   {
      m_helper.itemChanged(iItem);
   }
}

void
FWSimpleProxyBuilder::build(const FWEventItem* iItem,
                            TEveElementList* product, const FWViewContext* vc)
{
   size_t size = iItem->size();
   TEveElement::List_i pIdx = product->BeginChildren();
   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      TEveElement* itemHolder = 0;
      if (index <  product->NumChildren())
      {
         itemHolder = *pIdx;
         itemHolder->SetRnrSelfChildren(true, true);
         ++pIdx;
      }
      else
      {
         itemHolder = createCompound();
         product->AddElement(itemHolder);
      }
      if (iItem->modelInfo(index).displayProperties().isVisible())
      {
         const void* modelData = iItem->modelData(index);
         build(m_helper.offsetObject(modelData),index, *itemHolder, vc);
      }
   }
}

void
FWSimpleProxyBuilder::buildViewType(const FWEventItem* iItem,
                                    TEveElementList* product, FWViewType::EType viewType, const FWViewContext* vc)
{
   size_t size = iItem->size();
   TEveElement::List_i pIdx = product->BeginChildren();
   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      TEveElement* itemHolder = 0;
      if (index < product->NumChildren())
      {
         itemHolder = *pIdx;
         itemHolder->SetRnrSelfChildren(true, true);
         ++pIdx;
      }
      else
      {
         itemHolder = createCompound();
         product->AddElement(itemHolder);
      }
      if (iItem->modelInfo(index).displayProperties().isVisible())
      {
         const void* modelData = iItem->modelData(index);
         buildViewType(m_helper.offsetObject(modelData),index, *itemHolder, viewType, vc);
      }
   }
}


bool
FWSimpleProxyBuilder::visibilityModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                             FWViewType::EType viewType, const FWViewContext* vc)
{
   const FWEventItem::ModelInfo& info = iId.item()->modelInfo(iId.index());
   bool returnValue = false;
   if (info.displayProperties().isVisible() && iCompound->NumChildren()==0)
   {
      const void* modelData = iId.item()->modelData(iId.index());
      if (haveSingleProduct())      
         build(m_helper.offsetObject(modelData),iId.index(),*iCompound, vc);
      else
         buildViewType(m_helper.offsetObject(modelData),iId.index(),*iCompound, viewType, vc);
      returnValue=true;
   }
   return returnValue;
}



//
// const member functions
//

//
// static member functions
//
std::string
FWSimpleProxyBuilder::typeOfBuilder()
{
   return std::string("simple#");
}

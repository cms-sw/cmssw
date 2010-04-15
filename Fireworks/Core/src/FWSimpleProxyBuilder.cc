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
// $Id: FWSimpleProxyBuilder.cc,v 1.2 2010/04/09 10:58:24 amraktad Exp $
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "TEveCompound.h"
#include "TEveManager.h"
// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilder.h"
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
FWSimpleProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   if (iItem)
   {
      m_helper.itemChanged(iItem);
   }
}

void
FWSimpleProxyBuilder::build(const FWEventItem* iItem,
                            TEveElementList* product)
{
   size_t size = iItem->size();
   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      const void* modelData = iItem->modelData(index);
      std::string name;
      m_helper.fillTitle(*iItem,index,name);
      std::auto_ptr<TEveCompound> itemHolder(new TEveCompound(name.c_str(),name.c_str()));
      product->AddElement(itemHolder.get());
      const FWEventItem::ModelInfo& info = iItem->modelInfo(index);
      if (info.displayProperties().isVisible())
      {
         itemHolder->OpenCompound();
         //guarantees that CloseCompound will be called no matter what happens
         boost::shared_ptr<TEveCompound> sentry(itemHolder.get(),
                                                boost::mem_fn(&TEveCompound::CloseCompound));
         build(m_helper.offsetObject(modelData),index,*itemHolder); 
         changeElementAndChildren(itemHolder.get(), info);
      }
      itemHolder->SetRnrSelf(info.displayProperties().isVisible());
      itemHolder->SetRnrChildren(info.displayProperties().isVisible());
      itemHolder->ElementChanged();
      itemHolder.release();
   }
}

bool
FWSimpleProxyBuilder::specialModelChangeHandling(const FWModelId& iId, TEveElement* iCompound) {
   const FWEventItem::ModelInfo& info = iId.item()->modelInfo(iId.index());
   bool returnValue = false;
   if(info.displayProperties().isVisible() && iCompound->NumChildren()==0) {
      TEveCompound* c = dynamic_cast<TEveCompound*> (iCompound);
      assert(0!=c);
      c->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(c,
                                             boost::mem_fn(&TEveCompound::CloseCompound));
      const void* modelData = iId.item()->modelData(iId.index());
      
      build(m_helper.offsetObject(modelData),iId.index(),*iCompound);
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

// TODO ... handle new elements for new items => element->ProjectChild()

// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZSimpleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 19 12:39:36 EST 2008
// $Id: FWRPZSimpleProxyBuilder.cc,v 1.4 2008/12/02 21:17:46 chrjones Exp $
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "Reflex/Object.h"
#include "Reflex/Type.h"
#include "TClass.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
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
FWRPZSimpleProxyBuilder::FWRPZSimpleProxyBuilder(const std::type_info& iType) :
   m_containerPtr(new TEveElementList),
   m_helper(iType),
   m_needsUpdate(true)
{
}

// FWRPZSimpleProxyBuilder::FWRPZSimpleProxyBuilder(const FWRPZSimpleProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRPZSimpleProxyBuilder::~FWRPZSimpleProxyBuilder()
{
   m_containerPtr.destroyElement();
}

//
// assignment operators
//
// const FWRPZSimpleProxyBuilder& FWRPZSimpleProxyBuilder::operator=(const FWRPZSimpleProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWRPZSimpleProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWRPZSimpleProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   m_needsUpdate=true;
   m_helper.itemChanged(iItem);
   m_containerPtr->DestroyElements();
}

void
FWRPZSimpleProxyBuilder::itemBeingDestroyedImp(const FWEventItem*)
{
   m_needsUpdate=false;
   m_containerPtr->DestroyElements();
}

void
FWRPZSimpleProxyBuilder::modelChangesImp(const FWModelIds& iIds)
{
   modelChanges(iIds,m_containerPtr.get());
}

TEveElementList*
FWRPZSimpleProxyBuilder::getRhoPhiProduct() const
{
   if(m_needsUpdate) {
      m_needsUpdate=false;
      const_cast<FWRPZSimpleProxyBuilder*>(this)->build();
   }
   return m_containerPtr.get();
}

TEveElementList*
FWRPZSimpleProxyBuilder::getRhoZProduct() const
{
   if(m_needsUpdate) {
      m_needsUpdate=false;
      const_cast<FWRPZSimpleProxyBuilder*>(this)->build();
   }
   return m_containerPtr.get();
}

void
FWRPZSimpleProxyBuilder::build()
{
   if(0!=item()) {
      size_t size = item()->size();
      size_t largestIndex = ids().size();
      if(largestIndex < size) {
         ids().resize(size);
      }
      std::vector<FWModelId>::iterator itId = ids().begin();
      for(int index = 0; index < static_cast<int>(size); ++index,++itId) {
         if(index>=static_cast<int>(largestIndex)) {
            *itId=FWModelId(item(),index);
         }
         const void* modelData = item()->modelData(index);
         std::string name;
         m_helper.fillTitle(*item(), index, name);
         std::auto_ptr<TEveCompound> itemHolder(new TEveCompound(name.c_str(),name.c_str()));
         {
            itemHolder->OpenCompound();
            //guarantees that CloseCompound will be called no matter what happens
            boost::shared_ptr<TEveCompound> sentry(itemHolder.get(),
                                                   boost::mem_fn(&TEveCompound::CloseCompound));
            build(m_helper.offsetObject(modelData),index,*itemHolder);
         }
         const FWEventItem::ModelInfo& info = item()->modelInfo(index);
         changeElementAndChildren(itemHolder.get(), info);
         fireworks::setUserDataElementAndChildren(itemHolder.get(), static_cast<void*>(&(*itId)));
         itemHolder->SetRnrSelf(info.displayProperties().isVisible());
         itemHolder->SetRnrChildren(info.displayProperties().isVisible());
         itemHolder->ElementChanged();
         m_containerPtr->AddElement(itemHolder.release());
      }
   }
}

//
// const member functions
//

//
// static member functions
//

std::string
FWRPZSimpleProxyBuilder::typeOfBuilder()
{
   return std::string("simple#");
}

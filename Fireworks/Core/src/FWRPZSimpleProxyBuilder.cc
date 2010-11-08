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
// $Id: FWRPZSimpleProxyBuilder.cc,v 1.6 2009/10/28 14:46:17 chrjones Exp $
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>
#include <boost/bind.hpp>

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
   //Do I have to create the model?  Only do it if 
   // 1) model is visible
   // 2) model does not already exist
   
   //NOTE: life would be easier if I used a random access list rather than a TEveElementList
   TEveElement::List_i itElement = m_containerPtr->BeginChildren();
   int index = 0; //this keeps track of what index corresponds to itElement
   
   for(FWModelIds::const_iterator it = iIds.begin(), itEnd = iIds.end();
       it != itEnd;
       ++it,++itElement,++index) {
      
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(it->index());
      if(info.displayProperties().isVisible()) {
         assert(itElement != m_containerPtr->EndChildren());
         while(index < it->index()) {
            ++itElement;
            ++index;
            assert(itElement != m_containerPtr->EndChildren());
         }
         if((*itElement)->NumChildren()==0) {
            //now build them
            //std::cout <<"building "<<index<<std::endl;
            TEveCompound* itemHolder = dynamic_cast<TEveCompound*> (*itElement);
            itemHolder->OpenCompound();
            {
               //guarantees that CloseCompound will be called no matter what happens
               boost::shared_ptr<TEveCompound> sentry(itemHolder,
                                                      boost::mem_fn(&TEveCompound::CloseCompound));
               const void* modelData = it->item()->modelData(index);
               build(m_helper.offsetObject(modelData),index,*itemHolder);
            }
            changeElementAndChildren(itemHolder, info);
            fireworks::setUserDataElementAndChildren(itemHolder, static_cast<void*>(&(ids()[index])));
            //make sure this is 'projected'
            for(TEveElement::List_i itChild = itemHolder->BeginChildren(), itChildEnd = itemHolder->EndChildren();
                itChild != itChildEnd;
                ++itChild) {
               itemHolder->ProjectChild(*itChild);
               itemHolder->RecheckImpliedSelections();
            }
         }         
      }
   }
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
      std::vector<FWModelIdFromEveSelector>::iterator itId = ids().begin();
      for(int index = 0; index < static_cast<int>(size); ++index,++itId) {
         if(index>=static_cast<int>(largestIndex)) {
            *itId=FWModelId(item(),index);
         }
         const void* modelData = item()->modelData(index);
         std::string name;
         m_helper.fillTitle(*item(), index, name);
         std::auto_ptr<TEveCompound> itemHolder(new TEveCompound(name.c_str(),name.c_str()));
         const FWEventItem::ModelInfo& info = item()->modelInfo(index);
         if(info.displayProperties().isVisible()) {
            {
               itemHolder->OpenCompound();
               //guarantees that CloseCompound will be called no matter what happens
               boost::shared_ptr<TEveCompound> sentry(itemHolder.get(),
                                                      boost::mem_fn(&TEveCompound::CloseCompound));
               build(m_helper.offsetObject(modelData),index,*itemHolder);
            }
            changeElementAndChildren(itemHolder.get(), info);
            fireworks::setUserDataElementAndChildren(itemHolder.get(), static_cast<void*>(&(*itId)));
         }
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

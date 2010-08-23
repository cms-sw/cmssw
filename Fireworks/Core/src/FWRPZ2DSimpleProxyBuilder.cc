// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZ2DSimpleProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 11:27:21 EST 2008
// $Id: FWRPZ2DSimpleProxyBuilder.cc,v 1.6 2009/10/28 14:46:17 chrjones Exp $
//

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "Reflex/Type.h"
#include "Reflex/Base.h"
#include "Reflex/Object.h"
#include "TClass.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/setUserDataElementAndChildren.h"
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
FWRPZ2DSimpleProxyBuilder::FWRPZ2DSimpleProxyBuilder(const std::type_info& iType) :
   m_rhoPhiElementsPtr(new TEveElementList),
   m_rhoZElementsPtr(new TEveElementList),
   m_compounds(new TEveElementList),
   m_helper(iType),
   m_rhoPhiNeedsUpdate(true),
   m_rhoZNeedsUpdate(true)
{
}

// FWRPZ2DSimpleProxyBuilder::FWRPZ2DSimpleProxyBuilder(const FWRPZ2DSimpleProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRPZ2DSimpleProxyBuilder::~FWRPZ2DSimpleProxyBuilder()
{
   m_rhoPhiElementsPtr.destroyElement();
   m_rhoZElementsPtr.destroyElement();
   m_compounds.destroyElement();
}

//
// assignment operators
//
// const FWRPZ2DSimpleProxyBuilder& FWRPZ2DSimpleProxyBuilder::operator=(const FWRPZ2DSimpleProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWRPZ2DSimpleProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWRPZ2DSimpleProxyBuilder::itemChangedImp(const FWEventItem* iItem)
{
   m_rhoPhiNeedsUpdate=true;
   m_rhoZNeedsUpdate=true;
   m_helper.itemChanged(iItem);
   m_rhoPhiElementsPtr->DestroyElements();
   m_rhoZElementsPtr->DestroyElements();
}

void
FWRPZ2DSimpleProxyBuilder::itemBeingDestroyedImp(const FWEventItem*)
{
   m_rhoPhiNeedsUpdate=false;
   m_rhoZNeedsUpdate=false;
   m_rhoPhiElementsPtr->DestroyElements();
   m_rhoZElementsPtr->DestroyElements();
}

template<class TCaller>
void
FWRPZ2DSimpleProxyBuilder::buildIfNeeded(const FWModelIds& iIds, TEveElement* iParent)
{
   //Do I have to create the model?  Only do it if 
   // 1) model is visible
   // 2) model does not already exist
   
   //NOTE: life would be easier if I used a random access list rather than a TEveElementList
   TEveElement::List_i itElement = iParent->BeginChildren();
   int index = 0; //this keeps track of what index corresponds to itElement
   
   for(FWModelIds::const_iterator it = iIds.begin(), itEnd = iIds.end();
       it != itEnd;
       ++it,++itElement,++index) {
      
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(it->index());
      if(info.displayProperties().isVisible()) {
         assert(itElement != iParent->EndChildren());
         while(index < it->index()) {
            ++itElement;
            ++index;
            assert(itElement != iParent->EndChildren());
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
               TCaller::build(this,m_helper.offsetObject(modelData),index,*itemHolder);
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
}

class FWRPSimpleCaller {
public:
   static void build(FWRPZ2DSimpleProxyBuilder* iThis,
                     const void* iData,
                     unsigned int iIndex,
                     TEveElement& iItemHolder) {
      iThis->buildRhoPhi(iData,iIndex,iItemHolder);
   }
};

class FWRZSimpleCaller {
public:
   static void build(FWRPZ2DSimpleProxyBuilder* iThis,
                     const void* iData,
                     unsigned int iIndex,
                     TEveElement& iItemHolder) {
      iThis->buildRhoZ(iData,iIndex,iItemHolder);
   }

};

void
FWRPZ2DSimpleProxyBuilder::modelChangesImp(const FWModelIds& iIds)
{
   //If we need an update it means no view wanted this info and therefore
   // it was never made and therefore we should not try to update it
   if(!m_rhoPhiNeedsUpdate) {
      buildIfNeeded<FWRPSimpleCaller>(iIds,m_rhoPhiElementsPtr.get());
      modelChanges(iIds,m_rhoPhiElementsPtr.get());
   }
   if(!m_rhoZNeedsUpdate) {
      buildIfNeeded<FWRZSimpleCaller>(iIds,m_rhoZElementsPtr.get());
      modelChanges(iIds,m_rhoZElementsPtr.get());
   }
}



template<class T>
void
FWRPZ2DSimpleProxyBuilder::build(TEveElementList* oAddTo, T iCaller)
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
         m_helper.fillTitle(*item(),index,name);
         std::auto_ptr<TEveCompound> itemHolder(new TEveCompound(name.c_str(),name.c_str()));
         const FWEventItem::ModelInfo& info = item()->modelInfo(index);
         if(info.displayProperties().isVisible()) {
            {
               itemHolder->OpenCompound();
               //guarantees that CloseCompound will be called no matter what happens
               boost::shared_ptr<TEveCompound> sentry(itemHolder.get(),
                                                      boost::mem_fn(&TEveCompound::CloseCompound));
               iCaller.build(this,m_helper.offsetObject(modelData),index,*itemHolder);
            }
         }
         //NOTE: TEveCompounds only change the color of child elements which are already the
         // old color of the TEveCompound
         changeElementAndChildren(itemHolder.get(), info);
         //itemHolder->SetMainColor(info.displayProperties().color());
         fireworks::setUserDataElementAndChildren(itemHolder.get(), static_cast<void*>(&(*itId)));
         itemHolder->SetRnrSelf(info.displayProperties().isVisible());
         itemHolder->SetRnrChildren(info.displayProperties().isVisible());
         itemHolder->ElementChanged();
         oAddTo->AddElement(itemHolder.release());
      }
   }
}


TEveElementList*
FWRPZ2DSimpleProxyBuilder::getRhoPhiProduct() const
{
   if(m_rhoPhiNeedsUpdate) {
      m_rhoPhiNeedsUpdate=false;
      const_cast<FWRPZ2DSimpleProxyBuilder*>(this)->build(m_rhoPhiElementsPtr.get(),
                                                          FWRPSimpleCaller());
   }
   return m_rhoPhiElementsPtr.get();
}

TEveElementList*
FWRPZ2DSimpleProxyBuilder::getRhoZProduct() const
{
   if(m_rhoZNeedsUpdate) {
      m_rhoZNeedsUpdate=false;
      const_cast<FWRPZ2DSimpleProxyBuilder*>(this)->build(m_rhoZElementsPtr.get(),
                                                          FWRZSimpleCaller());
   }
   return m_rhoZElementsPtr.get();
}


//
// const member functions
//

//
// static member functions
//
std::string
FWRPZ2DSimpleProxyBuilder::typeOfBuilder()
{
   return std::string("simple#");
}

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
// $Id: FWRPZ2DSimpleProxyBuilder.cc,v 1.1 2008/11/27 00:40:34 chrjones Exp $
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
FWRPZ2DSimpleProxyBuilder::FWRPZ2DSimpleProxyBuilder(const std::type_info& iType):
m_rhoPhiElementsPtr(new TEveElementList),
m_rhoZElementsPtr(new TEveElementList),
m_compounds(new TEveElementList),
m_itemType(&iType),
m_objectOffset(0),
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
   if(0!=iItem) {
      using namespace ROOT::Reflex;
      Type myType = Type::ByTypeInfo(*m_itemType);
      Object dummy(Type::ByTypeInfo(*(iItem->modelType()->GetTypeInfo())),
                   reinterpret_cast<void*>(0xFFFF));
      Object castTo = dummy.CastObject(myType);
      assert(0!=castTo.Address());
      m_objectOffset=static_cast<char*>(dummy.Address())-static_cast<char*>(castTo.Address());
   }
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

void 
FWRPZ2DSimpleProxyBuilder::modelChangesImp(const FWModelIds& iIds)
{
   modelChanges(iIds,m_rhoPhiElementsPtr.get());
   modelChanges(iIds,m_rhoZElementsPtr.get());
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
      std::vector<FWModelId>::iterator itId = ids().begin();
      for(int index = 0; index < static_cast<int>(size); ++index,++itId) {
         if(index>=static_cast<int>(largestIndex)) {
            *itId=FWModelId(item(),index);
         }
         const void* modelData = item()->modelData(index);
         std::string name = item()->modelName(index);
         std::stringstream s;
         if(item()->haveInterestingValue()) {
            s<<name<<", "<<item()->modelInterestingValueAsString(index);
            name=s.str();
         }
         std::auto_ptr<TEveCompound> itemHolder(new TEveCompound(name.c_str(),name.c_str()));
         {
            itemHolder->OpenCompound();
            //guarantees that CloseCompound will be called no matter what happens
            boost::shared_ptr<TEveCompound> sentry(itemHolder.get(),
                                                   boost::mem_fn(&TEveCompound::CloseCompound));
            iCaller.build(this,static_cast<const char*>(modelData)+m_objectOffset,index,*itemHolder);
         }
         const FWEventItem::ModelInfo& info = item()->modelInfo(index);
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

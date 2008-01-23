// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FWRPZDataProxyBuilder.cc,v 1.2 2008/01/22 21:08:54 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWRPZDataProxyBuilder::FWRPZDataProxyBuilder():
  m_item(0)
{
}

// FWRPZDataProxyBuilder::FWRPZDataProxyBuilder(const FWRPZDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRPZDataProxyBuilder::~FWRPZDataProxyBuilder()
{
}

//
// assignment operators
//
// const FWRPZDataProxyBuilder& FWRPZDataProxyBuilder::operator=(const FWRPZDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWRPZDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWRPZDataProxyBuilder::setItem(const FWEventItem* iItem)
{
  m_item = iItem;
  if(0 != m_item) {
     m_item->changed_.connect(boost::bind(&FWRPZDataProxyBuilder::modelChanges,this,_1));
  }
}

void
FWRPZDataProxyBuilder::build(TEveElementList** iObject)
{
  if(0!= m_item) {
    build(m_item, iObject);
    m_elements=*iObject;
  }
}

void 
FWRPZDataProxyBuilder::modelChanges(const FWModelIds& iIds)
{
   modelChanges(iIds,m_elements);
}

static void
changeElementAndChildren(TEveElement* iElement, 
                         const FWEventItem::ModelInfo& iInfo)
{
   iElement->SetMainColor(iInfo.displayProperties().color());
   //for now, if selected make the item white
   if(iInfo.isSelected()) {
      iElement->SetMainColor(static_cast<Color_t>(kWhite));
   }

   for(TEveElement::List_i itElement = iElement->BeginChildren(),
       itEnd = iElement->EndChildren();
       itElement != itEnd;
       ++itElement) {
      changeElementAndChildren(*itElement, iInfo);
   }
}

void 
FWRPZDataProxyBuilder::modelChanges(const FWModelIds& iIds,
                                    TEveElementList* iElements )
{
   assert(m_item && m_item->size() == iElements->GetNChildren() && "can not use default modelChanges implementation");
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
   }
}

//
// const member functions
//

//
// static member functions
//

// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoDataProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec  6 17:49:54 PST 2007
// $Id: FW3DLegoDataProxyBuilder.cc,v 1.3.4.2 2008/03/18 01:40:04 dmytro Exp $
//

// system include files
#include <boost/bind.hpp>
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "TH2F.h"
#include "TEveElement.h"
#include <iostream>
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"

//
// constants, enums and typedefs
//
namespace fw3dlego
{
  const double xbins[83] = {
           -5.191, -4.889,
	   -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, 
	   -3.139, -2.964, -2.853, -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830, 
	   -1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, 
	   -0.870, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087,  
	   0.000,
	    0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.870,
	    0.957,  1.044,  1.131,  1.218,  1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  
	    1.830,  1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  2.853,  2.964,  3.139,  
	    3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716,
            4.889,  5.191
  };

}

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DLegoDataProxyBuilder::FW3DLegoDataProxyBuilder():
  m_item(0),
  m_elements(0)
{
}

// FW3DLegoDataProxyBuilder::FW3DLegoDataProxyBuilder(const FW3DLegoDataProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FW3DLegoDataProxyBuilder::~FW3DLegoDataProxyBuilder()
{
}

//
// assignment operators
//
// const FW3DLegoDataProxyBuilder& FW3DLegoDataProxyBuilder::operator=(const FW3DLegoDataProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FW3DLegoDataProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FW3DLegoDataProxyBuilder::setItem(const FWEventItem* iItem)
{
   m_item = iItem;
   if(0 != m_item) {
      m_item->changed_.connect(boost::bind(&FW3DLegoDataProxyBuilder::modelChanges,this,_1));
   }

}


static void
setUserDataElementAndChildren(TEveElement* iElement, 
                              void* iInfo)
{
   iElement->SetUserData(iInfo);
   for(TEveElement::List_i itElement = iElement->BeginChildren(),
       itEnd = iElement->EndChildren();
       itElement != itEnd;
       ++itElement) {
      setUserDataElementAndChildren(*itElement, iInfo);
   }
}

static
void
setUserData(const FWEventItem* iItem,TEveElementList* iElements, std::vector<FWModelId>& iIds) {
   if(iElements &&  static_cast<int>(iItem->size()) == iElements->GetNChildren() ) {
      int index=0;
      int largestIndex = iIds.size();
      if(iIds.size()<iItem->size()) {
         iIds.resize(iItem->size());
      }
      std::vector<FWModelId>::iterator itId = iIds.begin();
      for(TEveElement::List_i it = iElements->BeginChildren(), itEnd = iElements->EndChildren();
          it != itEnd;
          ++it,++itId,++index) {
         if(largestIndex<=index) {
            *itId=FWModelId(iItem,index);
         }
         setUserDataElementAndChildren(*it,&(*itId));
      }
   }
}


void
FW3DLegoDataProxyBuilder::build(TObject** iObject)
{
   if ( ! m_item ) return;
   TH2* hist = dynamic_cast<TH2*>(*iObject);
   if ( ! *iObject || hist ) build(m_item, &hist);
   if ( hist ) {
      *iObject = hist;
      return;
   }
   
   TEveElementList* list = dynamic_cast<TEveElementList*>(*iObject);
   if ( ! *iObject || list ) build(m_item, &list);
   if ( list ) {
      *iObject = list;
      m_elements = list;
      setUserData(m_item,list,m_ids);
   }
}

void 
FW3DLegoDataProxyBuilder::modelChanges(const FWModelIds& iIds)
{
   // printf("number of model ids: %d\n", iIds.size() );
   FW3DLegoDataProxyBuilder::modelChanges(iIds,m_elements);
}

static void
changeElementAndChildren(TEveElement* iElement, 
                         const FWEventItem::ModelInfo& iInfo)
{
   iElement->SetMainColor(iInfo.displayProperties().color());
   //for now, if selected make the item white
   if(iInfo.isSelected() xor iElement->GetSelectedLevel()==1) {
      if(iInfo.isSelected()) {         
         gEve->GetSelection()->AddElement(iElement);
      } else {
         gEve->GetSelection()->RemoveElement(iElement);
      }
   }

   for(TEveElement::List_i itElement = iElement->BeginChildren(),
       itEnd = iElement->EndChildren();
       itElement != itEnd;
       ++itElement) {
      changeElementAndChildren(*itElement, iInfo);
   }
}

void 
FW3DLegoDataProxyBuilder::modelChanges(const FWModelIds& iIds,
                                    TEveElement* iElements )
{
   //std::cout <<"modelChanged "<<m_item->size()<<" "<<iElements->GetNChildren()<<std::endl;
   if (! iElements) return;
   assert(m_item && static_cast<int>(m_item->size()) == iElements->GetNChildren() && "can not use default modelChanges implementation");
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
      (*itElement)->ElementChanged();
   }
}

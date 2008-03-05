// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListEventItem
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 28 11:13:37 PST 2008
// $Id: FWListEventItem.cc,v 1.4 2008/03/05 18:39:32 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>
#include <iostream>
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/FWListModel.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWListEventItem::FWListEventItem(FWEventItem* iItem,
                                 FWDetailViewManager* iDV):
TEveElementList(iItem->name().c_str(),"",kTRUE),
m_item(iItem),
m_detailViewManager(iDV)
{
   m_item->itemChanged_.connect(boost::bind(&FWListEventItem::itemChanged,this,_1));
   m_item->changed_.connect(boost::bind(&FWListEventItem::modelsChanged,this,_1));
   TEveElementList::SetMainColor(iItem->defaultDisplayProperties().color());
}

// FWListEventItem::FWListEventItem(const FWListEventItem& rhs)
// {
//    // do actual copying here;
// }

FWListEventItem::~FWListEventItem()
{
}

//
// assignment operators
//
// const FWListEventItem& FWListEventItem::operator=(const FWListEventItem& rhs)
// {
//   //An exception safe implementation is
//   FWListEventItem temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWListEventItem::SetMainColor(Color_t iColor)
{
   FWDisplayProperties prop(iColor,m_item->defaultDisplayProperties().isVisible());
   m_item->setDefaultDisplayProperties(prop);
   TEveElementList::SetMainColor(iColor);
}


void 
FWListEventItem::SetRnrState(Bool_t rnr)
{
   FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
   m_item->setDefaultDisplayProperties(prop);
   TEveElementList::SetRnrState(rnr);
   
}
Bool_t 
FWListEventItem::SingleRnrState() const
{
   return kTRUE;
}


void 
FWListEventItem::itemChanged(const FWEventItem* iItem)
{
   //std::cout <<"item changed "<<eventItem()->size()<<std::endl;
   this->DestroyElements();
   for(unsigned int index = 0; index < eventItem()->size(); ++index) {
      FWListModel* model = new FWListModel(FWModelId(eventItem(),index));
      this->AddElement( model );
      model->SetMainColor(m_item->defaultDisplayProperties().color());
   }
}

void 
FWListEventItem::modelsChanged( const std::set<FWModelId>& iModels )
{
   TEveElement::List_i itElement = this->BeginChildren();
   int index = 0;
   for(FWModelIds::const_iterator it = iModels.begin(), itEnd = iModels.end();
       it != itEnd;
       ++it,++itElement,++index) {
      assert(itElement != this->EndChildren());         
      while(index < it->index()) {
         ++itElement;
         ++index;
         assert(itElement != this->EndChildren());         
      }
      bool modelChanged = false;
      const FWEventItem::ModelInfo& info = it->item()->modelInfo(index);
      if((*itElement)->GetMainColor() != info.displayProperties().color() ) {
         (*itElement)->SetMainColor(info.displayProperties().color());
         modelChanged = true;
      }
      if(info.isSelected() xor (*itElement)->GetSelectedLevel()==1) {
         modelChanged = true;
         if(info.isSelected()) {         
            gEve->GetSelection()->AddElement(*itElement);
         } else {
            gEve->GetSelection()->RemoveElement(*itElement);
         }
      }
      
      if((*itElement)->GetRnrSelf() != info.displayProperties().isVisible()) {
         (*itElement)->SetRnrSelf(info.displayProperties().isVisible());
         modelChanged = true;
      }
      if(modelChanged) {
         (*itElement)->ElementChanged();
      }
   }
   
}

//
// const member functions
//
FWEventItem* 
FWListEventItem::eventItem() const
{
   return m_item;
}

void 
FWListEventItem::openDetailViewFor(int index) const
{
   m_detailViewManager->openDetailViewFor( FWModelId(m_item,index));
}

//
// static member functions
//

ClassImp(FWListEventItem)

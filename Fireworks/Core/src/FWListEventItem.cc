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
// $Id: FWListEventItem.cc,v 1.24 2008/12/01 01:00:57 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>
#include <iostream>
#include <sstream>
#include <map>
#include "TEveManager.h"
#include "TEveSelection.h"

#include "TClass.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "Reflex/Object.h"
#include "Reflex/Base.h"

// user include files
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/FWListModel.h"

#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"


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
                                 FWDetailViewManager* iDV) :
   TEveElementList(iItem->name().c_str(),"",kTRUE),
   m_item(iItem),
   m_detailViewManager(iDV)
{
   m_item->itemChanged_.connect(boost::bind(&FWListEventItem::itemChanged,this,_1));
   m_item->changed_.connect(boost::bind(&FWListEventItem::modelsChanged,this,_1));
   m_item->goingToBeDestroyed_.connect(boost::bind(&FWListEventItem::deleteListEventItem,this));
   m_item->defaultDisplayPropertiesChanged_.connect(boost::bind(&FWListEventItem::defaultDisplayPropertiesChanged,this,_1));
   TEveElementList::SetMainColor(iItem->defaultDisplayProperties().color());
   TEveElementList::SetRnrState(iItem->defaultDisplayProperties().isVisible());
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
FWListEventItem::deleteListEventItem() {
   delete this;
}

bool
FWListEventItem::doSelection(bool iToggleSelection)
{
   return true;
}

void
FWListEventItem::SetMainColor(Color_t iColor)
{
   FWDisplayProperties prop(iColor,m_item->defaultDisplayProperties().isVisible());
   m_item->setDefaultDisplayProperties(prop);
   TEveElementList::SetMainColor(iColor);
}


Bool_t
FWListEventItem::SetRnrState(Bool_t rnr)
{
   FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
   m_item->setDefaultDisplayProperties(prop);
   return TEveElementList::SetRnrState(rnr);
}
Bool_t
FWListEventItem::SingleRnrState() const
{
   return kTRUE;
}

void
FWListEventItem::defaultDisplayPropertiesChanged(const FWEventItem* iItem)
{
   TEveElementList::SetMainColor(iItem->defaultDisplayProperties().color());
   TEveElementList::SetRnrState(iItem->defaultDisplayProperties().isVisible());
}

void
FWListEventItem::itemChanged(const FWEventItem* iItem)
{
   //std::cout <<"item changed "<<iItem->name()<<std::endl;
   this->DestroyElements();
   m_indexOrderedItems.clear();
   m_indexOrderedItems.reserve(eventItem()->size());
   typedef std::multimap<double, FWListModel*, std::greater<double> > OrderedMap;
   OrderedMap orderedMap;
   if(iItem->isCollection() ) {
      for(unsigned int index = 0; index < eventItem()->size(); ++index) {
         ROOT::Reflex::Object obj;
         std::string data;
         double doubleData=index;
         if(eventItem()->haveInterestingValue()) {
            doubleData = eventItem()->modelInterestingValue(index);
            data = eventItem()->modelInterestingValueAsString(index);
         }
         FWListModel* model = new FWListModel(FWModelId(eventItem(),index),
                                              m_detailViewManager,
                                              data);
         m_indexOrderedItems.push_back(model);
         orderedMap.insert(std::make_pair(doubleData, model));
         //model->SetMainColor(m_item->defaultDisplayProperties().color());
         model->update(m_item->defaultDisplayProperties());
      }
      for(OrderedMap::iterator it = orderedMap.begin(), itEnd = orderedMap.end();
          it != itEnd;
          ++it) {
         this->AddElement( it->second );
      }
   }
}

void
FWListEventItem::modelsChanged( const std::set<FWModelId>& iModels )
{
   //std::cout <<"modelsChanged "<<std::endl;
   if(m_item->isCollection()) {
      bool aChildChanged = false;
      for(FWModelIds::const_iterator it = iModels.begin(), itEnd = iModels.end();
          it != itEnd;
          ++it) {
         int index = it->index();
         assert(index < static_cast<int>(m_indexOrderedItems.size()));
         FWListModel* element = m_indexOrderedItems[index];
         //std::cout <<"   "<<index<<std::endl;
         bool modelChanged = false;
         const FWEventItem::ModelInfo& info = it->item()->modelInfo(index);
         FWListModel* model = element;
         modelChanged = model->update(info.displayProperties());
         if(info.isSelected() xor element->GetSelectedLevel()==1) {
            modelChanged = true;
            if(info.isSelected()) {
               gEve->GetSelection()->AddElement(element);
            } else {
               gEve->GetSelection()->RemoveElement(element);
            }
         }
         if(modelChanged) {
            element->ElementChanged();
            aChildChanged=true;
            //(*itElement)->UpdateItems();  //needed to force list tree to update immediately
         }
      }
   }
   // obsolete
   // if(aChildChanged) {
   //    this->UpdateItems();
   // }
   //std::cout <<"modelsChanged done"<<std::endl;

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

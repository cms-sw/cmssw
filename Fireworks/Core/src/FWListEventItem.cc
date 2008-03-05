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
// $Id: FWListEventItem.cc,v 1.2 2008/03/05 15:13:50 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>
#include <iostream>

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
FWListEventItem::SetRnrChildren(Bool_t rnr)
{
   FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
   m_item->setDefaultDisplayProperties(prop);
   TEveElementList::SetRnrChildren(rnr);
   
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

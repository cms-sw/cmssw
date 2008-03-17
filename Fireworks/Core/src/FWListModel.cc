// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListModel
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar  3 17:20:14 EST 2008
// $Id: FWListModel.cc,v 1.8 2008/03/16 23:12:51 chrjones Exp $
//

// system include files
#include <assert.h>
#include <sstream>

// user include files
#include "Fireworks/Core/src/FWListModel.h"
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWListModel::FWListModel(const FWModelId& iId,
                         FWDetailViewManager* iDVM,
                         const std::string& iData):
TEveElement(m_color),
m_id(iId),
 m_detailViewManager(iDVM)
{
   std::ostringstream s;
   s<<m_id.index();
   if(not iData.empty()) {
      s<<": "<<iData;
   }
   SetElementName(s.str().c_str());
   SetUserData(&m_id);
}

// FWListModel::FWListModel(const FWListModel& rhs)
// {
//    // do actual copying here;
// }

FWListModel::~FWListModel()
{
}

//
// assignment operators
//
// const FWListModel& FWListModel::operator=(const FWListModel& rhs)
// {
//   //An exception safe implementation is
//   FWListModel temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
bool 
FWListModel::doSelection(bool iToggleSelection)
{
   if(iToggleSelection) {
      m_id.toggleSelect();
   } else {
      m_id.item()->selectionManager()->clearSelection();
      m_id.select();
   }
   return false;
}

bool 
FWListModel::update(const FWDisplayProperties& iProps)
{
   bool wasChanged=false;
   if(iProps.color() != GetMainColor()) {
      wasChanged = true;
      TEveElement::SetMainColor(iProps.color());
   }
   if(iProps.isVisible() != GetRnrState() ) {
      wasChanged=true;
      TEveElement::SetRnrState(iProps.isVisible());
   }
   return wasChanged;
}


void 
FWListModel::SetMainColor(Color_t iColor)
{
   const FWEventItem* item = m_id.item();
   FWDisplayProperties prop(iColor,item->modelInfo(m_id.index()).displayProperties().isVisible());
   item->setDisplayProperties(m_id.index(),prop);
   TEveElement::SetMainColor(iColor);
}


void 
FWListModel::SetRnrState(Bool_t rnr)
{
   //FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
   //m_item->setDefaultDisplayProperties(prop);
   const FWEventItem* item = m_id.item();
   FWDisplayProperties prop(item->modelInfo(m_id.index()).displayProperties().color(),rnr);
   item->setDisplayProperties(m_id.index(),prop);
   TEveElement::SetRnrState(rnr);   
}

Bool_t 
FWListModel::CanEditMainColor() const
{
   return kTRUE;
}

Bool_t 
FWListModel::SingleRnrState() const
{
   return kTRUE;
}

//
// const member functions
//
void 
FWListModel::openDetailView() const
{
     m_detailViewManager->openDetailViewFor(m_id);
}


//
// static member functions
//
ClassImp(FWListModel)

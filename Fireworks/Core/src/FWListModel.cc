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
// $Id$
//

// system include files
#include <assert.h>
#include <sstream>

// user include files
#include "Fireworks/Core/src/FWListModel.h"
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWListModel::FWListModel(int iIndex):
TEveElement(m_color),
m_index(iIndex)
{
   std::ostringstream s;
   s<<iIndex;
   SetElementName(s.str().c_str());
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
void 
FWListModel::SetMainColor(Color_t iColor)
{
   assert(1 == this->GetNParents());
   FWEventItem* item = static_cast<FWListEventItem*>(*(this->BeginParents()))->eventItem();
   FWDisplayProperties prop(iColor,item->modelInfo(m_index).displayProperties().isVisible());
   item->setDisplayProperties(m_index,prop);
   TEveElement::SetMainColor(iColor);
}


void 
FWListModel::SetRnrSelf(Bool_t rnr)
{
   //FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
   //m_item->setDefaultDisplayProperties(prop);
   assert(1 == this->GetNParents());
   FWEventItem* item = static_cast<FWListEventItem*>(*(this->BeginParents()))->eventItem();
   FWDisplayProperties prop(item->modelInfo(m_index).displayProperties().color(),rnr);
   item->setDisplayProperties(m_index,prop);
   TEveElement::SetRnrSelf(rnr);   
}

Bool_t 
FWListModel::CanEditMainColor() const
{
   return true;
}

//
// const member functions
//
void 
FWListModel::openDetailView() const
{
   
}


//
// static member functions
//
ClassImp(FWListModel)

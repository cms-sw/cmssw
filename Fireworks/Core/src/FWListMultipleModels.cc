// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListMultipleModels
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 24 11:45:18 EDT 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/src/FWListMultipleModels.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWListMultipleModels::FWListMultipleModels(const std::set<FWModelId>& iIDs):
TEveElement(m_color),
m_ids(iIDs)
{
   const FWEventItem* item = m_ids.begin()->item();
   TEveElement::SetMainColor(item->modelInfo(m_ids.begin()->index()).displayProperties().color());
   TEveElement::SetRnrState(item->modelInfo(m_ids.begin()->index()).displayProperties().isVisible());   
}

// FWListMultipleModels::FWListMultipleModels(const FWListMultipleModels& rhs)
// {
//    // do actual copying here;
// }

FWListMultipleModels::~FWListMultipleModels()
{
}

//
// assignment operators
//
// const FWListMultipleModels& FWListMultipleModels::operator=(const FWListMultipleModels& rhs)
// {
//   //An exception safe implementation is
//   FWListMultipleModels temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWListMultipleModels::SetMainColor(Color_t iColor)
{
   for(std::set<FWModelId>::iterator it = m_ids.begin(), itEnd=m_ids.end();
       it != itEnd;
       ++it) {
      const FWEventItem* item = it->item();
      FWDisplayProperties prop(iColor,item->modelInfo(it->index()).displayProperties().isVisible());
      item->setDisplayProperties(it->index(),prop);
      TEveElement::SetMainColor(iColor);
   }
}

void 
FWListMultipleModels::SetRnrState(Bool_t rnr)
{
   for(std::set<FWModelId>::iterator it = m_ids.begin(), itEnd=m_ids.end();
       it != itEnd;
       ++it) {
      //FWDisplayProperties prop(m_item->defaultDisplayProperties().color(),rnr);
      //m_item->setDefaultDisplayProperties(prop);
      const FWEventItem* item = it->item();
      FWDisplayProperties prop(item->modelInfo(it->index()).displayProperties().color(),rnr);
      item->setDisplayProperties(it->index(),prop);
      TEveElement::SetRnrState(rnr);   
   }
}

Bool_t 
FWListMultipleModels::CanEditMainColor() const
{
   return kTRUE;
}

Bool_t 
FWListMultipleModels::SingleRnrState() const
{
   return kTRUE;
}

bool 
FWListMultipleModels::doSelection(bool)
{
   return true;
}

//
// const member functions
//

//
// static member functions
//
ClassImp(FWListMultipleModels)

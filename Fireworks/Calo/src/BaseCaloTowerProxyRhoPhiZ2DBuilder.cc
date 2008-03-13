// -*- C++ -*-
//
// Package:     Calo
// Class  :     BaseCaloTowerProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Mar 12 21:32:54 CDT 2008
// $Id$
//

// system include files
#include "TEveElement.h"

// user include files
#include "Fireworks/Calo/interface/BaseCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/src/changeElementAndChildren.h"
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
BaseCaloTowerProxyRhoPhiZ2DBuilder::BaseCaloTowerProxyRhoPhiZ2DBuilder()
{
}

// BaseCaloTowerProxyRhoPhiZ2DBuilder::BaseCaloTowerProxyRhoPhiZ2DBuilder(const BaseCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

BaseCaloTowerProxyRhoPhiZ2DBuilder::~BaseCaloTowerProxyRhoPhiZ2DBuilder()
{
}

//
// assignment operators
//
// const BaseCaloTowerProxyRhoPhiZ2DBuilder& BaseCaloTowerProxyRhoPhiZ2DBuilder::operator=(const BaseCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//   //An exception safe implementation is
//   BaseCaloTowerProxyRhoPhiZ2DBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
BaseCaloTowerProxyRhoPhiZ2DBuilder::modelChangesRhoPhi(const FWModelIds& iIds, TEveElement* iElements)
{
   //for now, only if all items selected will will apply the action
   if(iIds.size() && iIds.size() == iIds.begin()->item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo& info = iIds.begin()->item()->modelInfo(iIds.begin()->index());
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();      
   }
}

void 
BaseCaloTowerProxyRhoPhiZ2DBuilder::modelChangesRhoZ(const FWModelIds& iIds, TEveElement* iElements)
{
   //for now, only if all items selected will will apply the action
   if(iIds.size() && iIds.size() == iIds.begin()->item()->size()) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo& info = iIds.begin()->item()->modelInfo(iIds.begin()->index());
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();      
   }
}

//
// const member functions
//

//
// static member functions
//

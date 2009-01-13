// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWBaseCaloTowerProxyRhoPhiZ2DBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Mar 12 21:32:54 CDT 2008
// $Id: FWBaseCaloTowerProxyRhoPhiZ2DBuilder.cc,v 1.1 2009/01/13 20:10:02 amraktad Exp $
//

// system include files
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

#include "Fireworks/Core/src/changeElementAndChildren.h"
#include "Fireworks/Core/interface/FWModelId.h"

class FWBaseCaloTowerProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      FWBaseCaloTowerProxyRhoPhiZ2DBuilder();
      virtual ~FWBaseCaloTowerProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
protected:
      //need to special handle selection or other changes
      virtual void modelChanges(const FWModelIds&, TEveElement*);
      virtual void applyChangesToAllModels(TEveElement*);


   private:
      FWBaseCaloTowerProxyRhoPhiZ2DBuilder(const FWBaseCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      const FWBaseCaloTowerProxyRhoPhiZ2DBuilder& operator=(const FWBaseCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------

};

//
// constructors and destructor
//
FWBaseCaloTowerProxyRhoPhiZ2DBuilder::FWBaseCaloTowerProxyRhoPhiZ2DBuilder()
{
}

// FWBaseCaloTowerProxyRhoPhiZ2DBuilder::FWBaseCaloTowerProxyRhoPhiZ2DBuilder(const FWBaseCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

FWBaseCaloTowerProxyRhoPhiZ2DBuilder::~FWBaseCaloTowerProxyRhoPhiZ2DBuilder()
{
}

//
// assignment operators
//
// const FWBaseCaloTowerProxyRhoPhiZ2DBuilder& FWBaseCaloTowerProxyRhoPhiZ2DBuilder::operator=(const FWBaseCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWBaseCaloTowerProxyRhoPhiZ2DBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWBaseCaloTowerProxyRhoPhiZ2DBuilder::modelChanges(const FWModelIds& iIds, TEveElement* iElements)
{
   //for now, only if all items selected will will apply the action
   if(iIds.size() && iIds.size() == iIds.begin()->item()->size()) {
      applyChangesToAllModels(iElements);
   }
}

void
FWBaseCaloTowerProxyRhoPhiZ2DBuilder::applyChangesToAllModels(TEveElement* iElements)
{
   if(ids().size() != 0 ) {
      //make the bad assumption that everything is being changed indentically
      const FWEventItem::ModelInfo& info = ids().begin()->item()->modelInfo(ids().begin()->index());
      changeElementAndChildren(iElements, info);
      iElements->SetRnrSelf(info.displayProperties().isVisible());
      iElements->SetRnrChildren(info.displayProperties().isVisible());
      iElements->ElementChanged();
   }
}

/*
void
FWBaseCaloTowerProxyRhoPhiZ2DBuilder::modelChangesRhoZ(const FWModelIds& iIds, TEveElement* iElements)
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
*/
//
// const member functions
//

//
// static member functions
//

// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDigitSetProxyBuilder
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Tue Oct 19 12:00:50 CEST 2010
// $Id: FWDigitSetProxyBuilder.cc,v 1.4 2012/06/18 23:56:21 amraktad Exp $
//

// system include files

// user include files
#include "TEveDigitSet.h"
#include "TEveBoxSet.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWFromEveSelectorBase.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"


class FWSecondarySelectableSelector : public FWFromEveSelectorBase
{
public:
   FWSecondarySelectableSelector(const TEveSecondarySelectable::SelectionSet_t& s, const FWEventItem* i): m_selected(s), m_item(i) {}
   ~FWSecondarySelectableSelector() {}

   virtual void doSelect()
   {
      syncSelection();
   }

   virtual void doUnselect()
   { 
      syncSelection(); 
   }

   const FWEventItem* item() const { return m_item; }

private:
   const TEveSecondarySelectable::SelectionSet_t& m_selected;
   const FWEventItem* m_item;

   void syncSelection()
   {
      size_t size = m_item->size();
      for (size_t i = 0; i < size; ++i)
      {
         FWEventItem::ModelInfo modelInfo = m_item->modelInfo(i);
         TEveSecondarySelectable::SelectionSet_ci si = m_selected.find(i);
         if ((si != m_selected.end()) != modelInfo.isSelected() )
         {
            if (si != m_selected.end())
               m_item->select(i);
            else
               m_item->unselect(i);
         }
      }
   }
};

//==============================================================================
//==============================================================================
//==============================================================================

FWDigitSetProxyBuilder::FWDigitSetProxyBuilder()
{
}

FWDigitSetProxyBuilder::~FWDigitSetProxyBuilder()
{
}

TString FWDigitSetProxyBuilder::getTooltip(TEveDigitSet* set, int idx)
{
   TEveElement* el = static_cast<TEveElement*>(set); // tmp-workaround
   FWSecondarySelectableSelector* ss = static_cast<FWSecondarySelectableSelector*>(el->GetUserData());
   return TString::Format("%d %s %s", idx, ss->item()->name().c_str(), ss->item()->modelInterestingValueAsString(idx).c_str());
}

TEveBoxSet* FWDigitSetProxyBuilder::addBoxSetToProduct(TEveElementList* product)
{
   assert(!product->HasChildren());
   
   TEveBoxSet* boxSet = new TEveBoxSet();
   boxSet->SetTooltipCBFoo(getTooltip);
   boxSet->Reset(TEveBoxSet::kBT_FreeBox, true, 256);
   FWSecondarySelectableSelector* sel = new FWSecondarySelectableSelector(boxSet->RefSelectedSet(), item());
   boxSet->SetUserData(sel);
   boxSet->SetPickable(1);
   boxSet->SetAlwaysSecSelect(1);

   product->AddElement(boxSet);

   return boxSet;
}

TEveDigitSet* FWDigitSetProxyBuilder::digitSet(TEveElement* product)
{
   assert(product->NumChildren() == 1);
   return static_cast<TEveDigitSet*>(*product->BeginChildren());
}

void FWDigitSetProxyBuilder::addBox(TEveBoxSet* boxSet, const float* pnts, const FWDisplayProperties& dp)
{
   boxSet->AddBox(pnts);
   boxSet->DigitValue(dp.isVisible());

   if (dp.isVisible()) 
      boxSet->DigitColor(dp.color(), dp.transparency());

   if (dp.transparency())
      boxSet->SetMainTransparency(dp.transparency());
}

void FWDigitSetProxyBuilder::modelChanges(const FWModelIds& iIds, Product* product)
{
   TEveDigitSet* digits = digitSet(product->m_elements);
   if (!digits) return;
   
   TEveSecondarySelectable::SelectionSet_t& selected = (TEveSecondarySelectable::SelectionSet_t&)(digits->RefSelectedSet());

   for (std::set<FWModelId>::const_iterator it = iIds.begin(); it != iIds.end(); ++it)
   {
      const FWEventItem::ModelInfo& info = item()->modelInfo(it->index());

      // id display properties
      const FWDisplayProperties &p = info.displayProperties();
      digits->SetCurrentDigit(it->index());
      digits->DigitValue(p.isVisible());
      if (p.isVisible())
         digits->DigitColor(p.color(), p.transparency());

      // id selection
      TEveSecondarySelectable::SelectionSet_ci si = selected.find(it->index());
      if (info.isSelected())
      {
         if (si == selected.end())
            selected.insert(it->index());
      }
      else
      {
         if ( si != selected.end())
            selected.erase(si);
      }
   }

   if(!selected.empty()) {
      if(0==digits->GetSelectedLevel()) {
         gEve->GetSelection()->AddElement(digits);
      }
   } else {
      if(1==digits->GetSelectedLevel()||2==digits->GetSelectedLevel()) {
         gEve->GetSelection()->RemoveElement(digits);
      }
   }

   digits->StampObjProps();
}

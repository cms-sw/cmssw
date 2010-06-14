// -*- C++ -*-
//
// Package:     Core
// Class  :     FWInteractionList
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Mon Apr 19 12:48:18 CEST 2010
// $Id: FWInteractionList.cc,v 1.7 2010/06/03 13:38:32 eulisse Exp $
//

// user include files
//#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "Fireworks/Core/interface/FWInteractionList.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelIdFromEveSelector.h"
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
FWInteractionList::FWInteractionList(const FWEventItem* item)
   : m_item(item)
{}

// FWInteractionList::FWInteractionList(const FWInteractionList& rhs)
// {
//    // do actual copying here;
// }

FWInteractionList::~FWInteractionList()
{
   for (size_t i = 0, e = m_compounds.size(); i != e; ++i)
   {
      TEveCompound *compound = m_compounds[i];
      // DOCREQ-GE: is there any case in which the compound does not contain
      //            the FWFromEveSelectorBase*? If so who deleted / removed it,
      //            since it was created in here? If by design the compound
      //            is always there the "if" should be changed to an assert, 
      //            IMHO.
      if (compound->GetUserData())
         delete reinterpret_cast<FWFromEveSelectorBase*>(compound->GetUserData());

      compound->RemoveElements();
      compound->DecDenyDestroy();
   }
}

//
// assignment operators
//
// const FWInteractionList& FWInteractionList::operator=(const FWInteractionList& rhs)
// {
//   //An exception safe implementation is
//   FWInteractionList temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
/** DOCREQ-GE: Is this a callback / slot?? When is this called??? */ 
void
FWInteractionList::added(TEveElement* el, unsigned int idx)
{
   // In the case a compound for the given index already exists, just add 
   // the TEveElement to it, otherwise create a new one.
   if (idx < m_compounds.size())
   {
      m_compounds[idx]->AddElement(el);
      return;
   }

   // DOCREQ-GE: what is the name used for?? Why do we keep the 
   //            "interestingValue" in the model name here?
   std::string name = m_item->modelName(idx);
   if (m_item->haveInterestingValue())
      name += ", " + m_item->modelInterestingValueAsString(idx);

   TEveCompound* c = new TEveCompound(name.c_str(), name.c_str());
   c->EnableListElements(m_item->defaultDisplayProperties().isVisible());
   c->SetMainColor(m_item->defaultDisplayProperties().color());
   // GE: Notice I added the following SetMainTransparency() call, while
   //     trying to get transparency to work on configuration load / reload.
   c->SetMainTransparency(m_item->defaultDisplayProperties().transparency());
   
   // DOCREQ-GE: I assume these are accessors for given flags, not actual
   //            actions, no? Otherwise, AFAICT they get applied on an empty 
   //            set of children. If so, shouldn't they be called "Set*" for 
   //            clarity?
   c->CSCImplySelectAllChildren();
   c->CSCApplyMainColorToAllChildren();
   c->CSCApplyMainTransparencyToAllChildren();

   // DOCREQ-GE: this is reference counting, I guess. How about renaming it to
   //            Ref / Unref? Shouldn't this done as first thing / last thing
   //            to avoid thinking that the ordering matters?
   c->IncDenyDestroy();
   // DOCREQ-GE: why is the FWModelIdFromEveSelector stored in the compound?
   //            Who uses this information?
   c->SetUserData(new FWModelIdFromEveSelector(FWModelId(m_item, idx)));
   // DOCREQ-GE: Does the ordering really matter here? Why is the element added
   //            after the new compound is pushed to the list???
   //            If not, we should probably do the push_back at the end to avoid
   //            the doubt.
   m_compounds.push_back(c);
   m_compounds.back()->AddElement(el); 
   // printf("%s[%d] FWInteractionList::added has childern %d\n",m_item->name().c_str(), idx,  m_compounds[idx]->NumChildren()); 
}

/** DOCREQ-GE: Is this a callback? When is this called? */
void
FWInteractionList::removed(TEveElement* el, int idx)
{
   m_compounds[idx]->RemoveElement(el);
}

/** DOCREQ-GE: When is this called? By who? */
void
FWInteractionList::modelChanges(const FWModelIds& iIds)
{ 
   assert (m_compounds.size() >= m_item->size());

   for (std::set<FWModelId>::const_iterator it = iIds.begin(); it != iIds.end(); ++it)
   {
      const FWEventItem::ModelInfo& info = m_item->modelInfo(it->index());
      // std::cout <<" FWInteractionList::modelChanges  color "<< info.displayProperties().color()  << "(*it).index() " <<(*it).index() << "  " << m_item->name() <<std::endl;
      const FWDisplayProperties &p = info.displayProperties();
      TEveElement* comp = m_compounds[it->index()];
      comp->EnableListElements(p.isVisible(), p.isVisible());
      comp->SetMainColor(p.color());
      comp->SetMainTransparency(p.transparency());

      if (info.isSelected())
      {
         if (comp->GetSelectedLevel() != 1)
            gEve->GetSelection()->AddElement(comp);
      }
      else
      {
         if (comp->GetSelectedLevel() == 1)
            gEve->GetSelection()->RemoveElement(comp);
      }
   }
}

/** DOCREQ-GE: When is this called? By who?*/
void
FWInteractionList::itemChanged()
{
   for (size_t i = 0, e = m_item->size(); i < e; ++i)
   {
      // DOCREQ-GE: Why are we assuming that m_item has the same size as 
      //            m_compounds?
      TEveElement* comp = m_compounds[i];
      
      std::string name = m_item->modelName(i);
      if (m_item->haveInterestingValue())
         name += ", " + m_item->modelInterestingValueAsString(i);

      comp->SetElementTitle(name.c_str());

      const FWEventItem::ModelInfo& info = m_item->modelInfo(i);
      const FWDisplayProperties &p = info.displayProperties();
      comp->EnableListElements(p.isVisible(), p.isVisible());
      comp->SetMainColor(p.color());
      comp->SetMainTransparency(p.transparency());
   }
}

//
// const member functions
//

//
// static member functions
//

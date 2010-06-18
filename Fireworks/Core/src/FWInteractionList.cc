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
// $Id: FWInteractionList.cc,v 1.9 2010/06/14 14:44:42 amraktad Exp $
//

// user include files
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
      // AT:
      // Interaction are created only in the standard use case, where user data is FWFromEveSelectorBase.
      // This is defined with return value of virtual function FWPRoxyBuilderBase::willHandleInteraction().

      if (compound->GetUserData())
         delete reinterpret_cast<FWFromEveSelectorBase*>(compound->GetUserData());

      compound->RemoveElements();
      compound->DecDenyDestroy();
   }
}


//
// member functions
//

void
FWInteractionList::added(TEveElement* el, unsigned int idx)
{
   /* DOCREQ-GE: Is this a callback  / slot?? When is this called??? */ 
   // AT:
   // This function is called from FWProxyBuilderBase::build() function (e.g. on next event).
   // The PB build function creates TEveElement for each element of collection and calls
   // this function to add the element to "master" element, which is a TEveCompound.

   // In the case a compound for the given index already exists, just add 
   // the TEveElement to it, otherwise create a new one.
   if (idx < m_compounds.size())
   {
      m_compounds[idx]->AddElement(el);
      return;
   }

   // DOCREQ-GE: what is the name used for?? Why do we keep the 
   //            "interestingValue" in the model name here?
   // AT:
   // Prepare name for the tooltip on mouseover in GL viewer.Value of
   // tooltip is TEveElement::fTitle
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
   // AT:set would be a better name, agree
   // Set flags to propagat attributes.
   c->CSCImplySelectAllChildren();
   c->CSCApplyMainColorToAllChildren();
   c->CSCApplyMainTransparencyToAllChildren();

   // DOCREQ-GE: this is reference counting, I guess. How about renaming it to
   //            Ref / Unref? Shouldn't this done as first thing / last thing
   //            to avoid thinking that the ordering matters?
   // AT:
   // TEveElement is auto-destroyed if is is not added to any parent. Alternative could 
   // be to use increase/decrease reference count.
   c->IncDenyDestroy();
   // DOCREQ-GE: why is the FWModelIdFromEveSelector stored in the compound?
   //            Who uses this information?
   //  AT:
   //  FWModelIdFromEveSelector is needed for interaction from Eve to Fireworks.
   //  FWEveViewManager gets ROOT signals with selected objects (TEveCompound)
   //  then cals doSelect() on the compound's user data.
   c->SetUserData(new FWModelIdFromEveSelector(FWModelId(m_item, idx)));
   // DOCREQ-GE: Does the ordering really matter here? Why is the element added
   //            after the new compound is pushed to the list???
   //            If not, we should probably do the push_back at the end to avoid
   //            the doubt.
   // AT:
   // Order does not matter. What is added to TEveCompound is not concern of interaction list.
   // Interaction list operates ony with the compound.
   m_compounds.push_back(c);
   m_compounds.back()->AddElement(el); 
   // printf("%s[%d] FWInteractionList::added has childern %d\n",m_item->name().c_str(), idx,  m_compounds[idx]->NumChildren()); 
}

/*
// Function removed() is not needed. Proxy builder takes care of product destructions.
// TEveElemnt::Destry() removes the element from all parents.
void
FWInteractionList::removed(TEveElement* el, int idx)
{
   m_compounds[idx]->RemoveElement(el);
}
*/

void
FWInteractionList::modelChanges(const FWModelIds& iIds)
{ 
   /** DOCREQ-GE: When is this called? By who? */
   // AT:
   // This method is called from FWEveViewManager::modelChanges(), which
   // has modelChanges callback same as all other view maangers.

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


void
FWInteractionList::itemChanged()
{
   /** DOCREQ-GE: When is this called? By who?*/
   // AT:
   // This method is called from FWEveViewManager::itemChanged(), which is a callback of
   // signal FWEventItem::itemChanged_.
   for (size_t i = 0, e = m_item->size(); i < e; ++i)
   {
      // DOCREQ-GE: Why are we assuming that m_item has the same size as 
      //            m_compounds?
      // AT: 
      // Assert for sizes is not necessary, becuse it is already in a 
      // proxy builder.
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

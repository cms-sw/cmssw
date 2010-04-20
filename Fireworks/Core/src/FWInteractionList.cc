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
// $Id$
//

// system include files
#include <sstream>

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
{
   m_item = item;
}

// FWInteractionList::FWInteractionList(const FWInteractionList& rhs)
// {
//    // do actual copying here;
// }

FWInteractionList::~FWInteractionList()
{
   for (std::vector<TEveCompound*>::iterator i = m_compounds.begin(); i!= m_compounds.end(); ++i)
   {
      if ((*i)->GetUserData())
      {
         FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*> ((*i)->GetUserData());
         delete base;
      }
      (*i)->RemoveElements();
      (*i)->DecDenyDestroy();
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
void
FWInteractionList::added(TEveElement* el, unsigned int idx)
{
   if (idx + 1 > m_compounds.size())
   {
      m_compounds.resize(idx+1);

      std::string name = m_item->modelName(idx);
      std::stringstream s;
      if (m_item->haveInterestingValue())
      {
         s<<name<<", "<<m_item->modelInterestingValueAsString(idx);
         name = s.str();
      }

      TEveCompound* c = new TEveCompound(name.c_str(),name.c_str());
      c->EnableListElements((m_item->defaultDisplayProperties().isVisible()));
      c->SetMainColor(m_item->defaultDisplayProperties().color());
      //   c->OpenCompound();
      c->CSCImplySelectAllChildren();
      c->CSCApplyMainColorToAllChildren();

      c->IncDenyDestroy();
      c->SetUserData(new FWModelIdFromEveSelector(FWModelId(m_item,idx)));
      m_compounds[idx] = c;
   }
   m_compounds[idx]->AddElement(el); 
   //  printf("%s[%d] FWInteractionList::added has childern %d\n",m_item->name().c_str(), idx,  m_compounds[idx]->NumChildren()); 

}

void
FWInteractionList::removed(TEveElement* el, int idx)
{
   m_compounds[idx]->RemoveElement(el);
}

void
FWInteractionList::modelChanges(const FWModelIds& iIds)
{ 
   if (m_compounds.size() != m_item->size())
   {
      printf("%s (m_compounds.size() != m_item->size() : %d/%d \n", m_item->name().c_str(), m_compounds.size(), m_item->size());
      return;
   }   
   

   for (std::set<FWModelId>::const_iterator it = iIds.begin(); it != iIds.end(); ++it)
   {
      const FWEventItem::ModelInfo& info = m_item->modelInfo((*it).index());
      // std::cout <<" FWInteractionList::modelChanges  color "<< info.displayProperties().color()  << "(*it).index() " <<(*it).index() << "  " << m_item->name() <<std::endl;
   
      TEveElement* comp = m_compounds[(*it).index()];
      comp->EnableListElements(info.displayProperties().isVisible(), info.displayProperties().isVisible());
      comp->SetMainColor(info.displayProperties().color());

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

//
// const member functions
//

//
// static member functions
//

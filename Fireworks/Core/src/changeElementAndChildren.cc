// -*- C++ -*-
//
// Package:     Core
// Class  :     changeElementAndChildren
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 11 21:41:15 CDT 2008
// $Id: changeElementAndChildren.cc,v 1.1 2008/03/12 02:53:25 chrjones Exp $
//

// system include files
#include "TEveProjectionBases.h"
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Core/src/changeElementAndChildren.h"

void
changeElementAndChildren(TEveElement* iElement,
                         const FWEventItem::ModelInfo& iInfo)
{
   iElement->SetMainColor(iInfo.displayProperties().color());
   //for now, if selected make the item white
   //std::cout <<"changeElementAndChildren "<<iElement <<" level "<<static_cast<int>(iElement->GetSelectedLevel())<<std::endl;
   if(iInfo.isSelected() xor (iElement->GetSelectedLevel()==1 or iElement->GetSelectedLevel()==2)) {
      if(iInfo.isSelected()) {
         gEve->GetSelection()->AddElement(iElement);
      } else {
         if(iElement->GetSelectedLevel()==1) {
            gEve->GetSelection()->RemoveElement(iElement);
         } else {
            //we are dealing with a selection level of 2 which means it was
            // implied selected, i.e. the Projectable was selected and therefore the
            // projected was 'implied' selected.  This means we must find the original
            // Projectable and stop it from being selected
            TEveProjected* pb = dynamic_cast<TEveProjected*>(iElement);
            if(0!=pb){
               TEveProjectable* pable = pb->GetProjectable();
               TEveElement* pAbleElement = dynamic_cast<TEveElement*>(pable);
               if(pAbleElement->GetSelectedLevel()==1) {
                  gEve->GetSelection()->RemoveElement(pAbleElement);
               }
            }
         }
         //std::cout <<"removing "<<iElement<<std::endl;
      }
   }

   for(TEveElement::List_i itElement = iElement->BeginChildren(),
       itEnd = iElement->EndChildren();
       itElement != itEnd;
       ++itElement) {
      changeElementAndChildren(*itElement, iInfo);
   }
}

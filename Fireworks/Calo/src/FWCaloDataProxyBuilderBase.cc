// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloDataProxyBuilderBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 15:09:39 CEST 2010
// $Id: FWCaloDataProxyBuilderBase.cc,v 1.5 2010/11/09 16:56:23 amraktad Exp $
//

// system include files

#include <math.h>

// user includes
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Calo/interface/FWCaloDataProxyBuilderBase.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCaloDataProxyBuilderBase::FWCaloDataProxyBuilderBase() :
   m_caloData(0),
   m_sliceIndex(-1)
{
}

// FWCaloDataProxyBuilderBase::FWCaloDataProxyBuilderBase(const FWCaloDataProxyBuilderBase& rhs)
// {
//    // do actual copying here;
// }

FWCaloDataProxyBuilderBase::~FWCaloDataProxyBuilderBase()
{
}

//
// assignment operators
//
// const FWCaloDataProxyBuilderBase& FWCaloDataProxyBuilderBase::operator=(const FWCaloDataProxyBuilderBase& rhs)
// {
//   //An exception safe implementation is
//   FWCaloDataProxyBuilderBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWCaloDataProxyBuilderBase::build(const FWEventItem* iItem,
                                  TEveElementList*, const FWViewContext*)
{
   setCaloData(iItem->context());

   assertCaloDataSlice();
   fillCaloData();

   m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
   m_caloData->SetSliceTransparency(m_sliceIndex,item()->defaultDisplayProperties().transparency());
   m_caloData->DataChanged();
   m_caloData->CellSelectionChanged();
}

//______________________________________________________________________________

void
FWCaloDataProxyBuilderBase::modelChanges(const FWModelIds&, Product* p)
{
   if(m_caloData && item())
   {      
      clearCaloDataSelection();
      fillCaloData();    

      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();
      if(!selected.empty()) {
         if(0==m_caloData->GetSelectedLevel()) {
            gEve->GetSelection()->AddElement(m_caloData);
         }
      } else {
         if(1==m_caloData->GetSelectedLevel()||2==m_caloData->GetSelectedLevel()) {
            gEve->GetSelection()->RemoveElement(m_caloData);
         }
      }   

      m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      m_caloData->SetSliceTransparency(m_sliceIndex,item()->defaultDisplayProperties().transparency());
      m_caloData->DataChanged();
      m_caloData->CellSelectionChanged();
   }
}
//______________________________________________________________________________

void
FWCaloDataProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   FWProxyBuilderBase::itemBeingDestroyed(iItem);
   if (m_caloData)
   {
      clearCaloDataSelection();
      FWFromTEveCaloDataSelector* sel = reinterpret_cast<FWFromTEveCaloDataSelector*>(m_caloData->GetUserData());
      sel->resetSliceSelector(m_sliceIndex);
      m_caloData->DataChanged();
   }
}



void
FWCaloDataProxyBuilderBase::clearCaloDataSelection()
{
   //find all selected cell ids which are not from this FWEventItem and preserve only them
   // do this by moving them to the end of the list and then clearing only the end of the list
   // this avoids needing any additional memory

   TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

   TEveCaloData::vCellId_t::iterator itEnd = selected.end();
   for(TEveCaloData::vCellId_t::iterator it = selected.begin();
       it != itEnd;
       ++it) {
      if(it->fSlice ==m_sliceIndex) {
         //we have found one we want to get rid of, so we swap it with the
         // one closest to the end which is not of this slice
         do {
            TEveCaloData::vCellId_t::iterator itLast = itEnd-1;
            itEnd = itLast;
         } while (itEnd != it && itEnd->fSlice==m_sliceIndex);
            
         if(itEnd != it) {
            std::swap(*it,*itEnd);
         } else {
            //shouldn't go on since advancing 'it' will put us past itEnd
            break;
         }
         //std::cout <<"keeping "<<it->fTower<<" "<<it->fSlice<<std::endl;
      }
   }
   selected.erase(itEnd,selected.end());


   // reset higlight
   m_caloData->GetCellsHighlighted().clear();
}

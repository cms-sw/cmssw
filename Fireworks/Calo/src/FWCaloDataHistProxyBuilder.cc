// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloDataHistProxyBuilder
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 15:09:39 CEST 2010
// $Id: FWCaloDataHistProxyBuilder.cc,v 1.2 2010/06/01 10:16:40 amraktad Exp $
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
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

#include "Fireworks/Calo/interface/FWCaloDataHistProxyBuilder.h"
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
FWCaloDataHistProxyBuilder::FWCaloDataHistProxyBuilder() :
   m_caloData(0),
   m_hist(0)
{
}

// FWCaloDataHistProxyBuilder::FWCaloDataHistProxyBuilder(const FWCaloDataHistProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWCaloDataHistProxyBuilder::~FWCaloDataHistProxyBuilder()
{
}

//
// assignment operators
//
// const FWCaloDataHistProxyBuilder& FWCaloDataHistProxyBuilder::operator=(const FWCaloDataHistProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWCaloDataHistProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWCaloDataHistProxyBuilder::build(const FWEventItem* iItem,
                                  TEveElementList*, const FWViewContext*)
{
   setCaloData(iItem->context());

   if(!item()) {
      if(0!=m_hist) {
         m_hist->Reset();
         m_caloData->DataChanged();
      }
      return;
   }

   assertHistogram();
   fillCaloData();

   m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
   m_caloData->DataChanged();
   m_caloData->CellSelectionChanged();
}

//______________________________________________________________________________

void
FWCaloDataHistProxyBuilder::modelChanges(const FWModelIds&, Product* p)
{
   applyChangesToAllModels(p);
}

void
FWCaloDataHistProxyBuilder::applyChangesToAllModels(Product* p)
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
      m_caloData->DataChanged();
      m_caloData->CellSelectionChanged();
   }
}

//______________________________________________________________________________

void
FWCaloDataHistProxyBuilder::itemBeingDestroyed(const FWEventItem* iItem)
{
   FWProxyBuilderBase::itemBeingDestroyed(iItem);
   if(0!=m_hist) 
   {
      clearCaloDataSelection();
      m_hist->Reset();
   }

   FWFromTEveCaloDataSelector* sel = reinterpret_cast<FWFromTEveCaloDataSelector*>(iItem->context().getCaloData()->GetUserData());
   sel->resetSliceSelector(m_sliceIndex);

   if(m_caloData) {
      m_caloData->DataChanged();
   }
}

//______________________________________________________________________________
bool
FWCaloDataHistProxyBuilder::assertHistogram()
{
   if (m_hist == 0)
   {
      // add new slice
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      m_hist = new TH2F(histName().c_str(),
                        histName().c_str(),
                        82, fw3dlego::xbins,
                        72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      m_sliceIndex = m_caloData->AddHistogram(m_hist);
      m_caloData->RefSliceInfo(m_sliceIndex).Setup(histName().c_str(), 0., item()->defaultDisplayProperties().color());

      // add new selector
      FWFromTEveCaloDataSelector* sel = 0;
      if (m_caloData->GetUserData())
      {
         FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(m_caloData->GetUserData());
         assert(0!=base);
         sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
         assert(0!=sel);
      }
      else
      {
         sel = new FWFromTEveCaloDataSelector(m_caloData);
         //make sure it is accessible via the base class
         m_caloData->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
      }

      addSliceSelector();   

      return true;
   }
   return false;
}

void
FWCaloDataHistProxyBuilder::clearCaloDataSelection()
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
}

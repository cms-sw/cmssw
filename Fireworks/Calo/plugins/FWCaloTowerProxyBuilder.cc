// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTowerProxyBuilderBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:28 EST 2008
// $Id: FWCaloTowerProxyBuilder.cc,v 1.11 2010/05/10 11:49:40 amraktad Exp $
//

// system includes
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

#include "Fireworks/Calo/plugins/FWCaloTowerProxyBuilder.h"
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"

//
// constructors , dectructors
//
FWCaloTowerProxyBuilderBase::FWCaloTowerProxyBuilderBase() :
   m_caloData(0),
   m_hist(0)
{
}

FWCaloTowerProxyBuilderBase::~FWCaloTowerProxyBuilderBase()
{
}
 
//
// member functions
//


void
FWCaloTowerProxyBuilderBase::build(const FWEventItem* iItem,
                                   TEveElementList*, const FWViewContext*)
{
   m_towers=0;
   iItem->get(m_towers);
   m_caloData = context().getCaloData();

   if(0==m_towers) {
      if(0!=m_hist) {
         m_hist->Reset();
         m_caloData->DataChanged();
      }
      return;
   }

   if(0==m_hist) {
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      m_hist = new TH2F(histName().c_str(),
                        (std::string("CaloTower ")+histName()+" Et distribution").c_str(),
                        82, fw3dlego::xbins,
                        72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      m_sliceIndex = m_caloData->AddHistogram(m_hist);
      m_caloData->RefSliceInfo(m_sliceIndex).Setup(histName().c_str(), 0., iItem->defaultDisplayProperties().color());

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

      sel->addSliceSelector(m_sliceIndex,FWFromSliceSelector(m_hist,iItem));      
   }
   m_hist->Reset();


   if(iItem->defaultDisplayProperties().isVisible()) {
      m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower) {
         (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
      }
   }
   m_caloData->DataChanged();
}

void
FWCaloTowerProxyBuilderBase::modelChanges(const FWModelIds&, Product* p)
{
   applyChangesToAllModels(p);
}

void
FWCaloTowerProxyBuilderBase::applyChangesToAllModels(Product* p)
{
   if(m_caloData && m_towers && item()) {
      m_hist->Reset();

      clearCaloDataSelection();

      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
            }
            if(info.isSelected()) {
               //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
               // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice
               // printf("applyChangesToAllModels ...check selected \n");
               selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(),tower->phi()),m_sliceIndex));
            }
         }
      }
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
   }
}

void
FWCaloTowerProxyBuilderBase::clearCaloDataSelection()
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


void
FWCaloTowerProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
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

REGISTER_FWPROXYBUILDER(FWECalCaloTowerProxyBuilder,CaloTowerCollection,"ECal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kLegoBit);
REGISTER_FWPROXYBUILDER(FWHCalCaloTowerProxyBuilder,CaloTowerCollection,"HCal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kLegoBit );


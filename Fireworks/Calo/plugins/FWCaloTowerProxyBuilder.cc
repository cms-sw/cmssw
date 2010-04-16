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
// $Id: FWCaloTowerProxyBuilder.cc,v 1.3 2010/04/15 20:15:14 amraktad Exp $
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
                                   TEveElementList* /*product*/)
{
   m_towers=0;
   iItem->get(m_towers);


   if(0==m_towers) {
      if(0!=m_hist) {
         m_hist->Reset();
         caloData()->DataChanged();
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
      m_sliceIndex = caloData()->AddHistogram(m_hist);
      caloData()->RefSliceInfo(m_sliceIndex).Setup(histName().c_str(), 0., iItem->defaultDisplayProperties().color());

      FWFromTEveCaloDataSelector* sel = 0;
      if (caloData()->GetUserData())
      {
         FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(caloData()->GetUserData());
         assert(0!=base);
         sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
         assert(0!=sel);
      }
      else
      {
         sel = new FWFromTEveCaloDataSelector(caloData());
         //make sure it is accessible via the base class
         caloData()->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
      }

      sel->addSliceSelector(m_sliceIndex,FWFromSliceSelector(m_hist,iItem));      
   }
   m_hist->Reset();


   if(iItem->defaultDisplayProperties().isVisible()) {
      caloData()->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower) {
         (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
      }
   }
   caloData()->DataChanged();
}

void
FWCaloTowerProxyBuilderBase::modelChanges(const FWModelIds&, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
FWCaloTowerProxyBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   if(caloData() && m_towers && item()) {
      m_hist->Reset();

      //find all selected cell ids which are not from this FWEventItem and preserve only them
      // do this by moving them to the end of the list and then clearing only the end of the list
      // this avoids needing any additional memory
      TEveCaloData::vCellId_t& selected = caloData()->GetCellsSelected();
      
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
               
               selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(),tower->phi()),m_sliceIndex));
            }
         }
      }

      if(!selected.empty()) {
         if(0==caloData()->GetSelectedLevel()) {
            gEve->GetSelection()->AddElement(caloData());
         }
      } else {
         if(1==caloData()->GetSelectedLevel()||2==caloData()->GetSelectedLevel()) {
            gEve->GetSelection()->RemoveElement(caloData());
         }
      }
      
      caloData()->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      caloData()->CellSelectionChanged();
      caloData()->DataChanged();

   }
}

void
FWCaloTowerProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   FWProxyBuilderBase::itemBeingDestroyed(iItem);
   if(0!=m_hist) {
      m_hist->Reset();
   }
   if(0 != caloData()) {
      caloData()->DataChanged();
   }
}

TEveCaloDataHist*
FWCaloTowerProxyBuilderBase::caloData() const
{
   return context().getCaloData();
}

REGISTER_FWPROXYBUILDER(FWECalCaloTowerProxyBuilder,CaloTowerCollection,"ECal",FWViewType::k3DBit|FWViewType::kRPZBit|FWViewType::kLegoBit);
REGISTER_FWPROXYBUILDER(FWHCalCaloTowerProxyBuilder,CaloTowerCollection,"HCal",FWViewType::k3DBit|FWViewType::kRPZBit|FWViewType::kLegoBit );


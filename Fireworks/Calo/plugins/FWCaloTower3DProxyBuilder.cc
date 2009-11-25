// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTower3DProxyBuilderBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Dec  3 11:28:28 EST 2008
// $Id: FWCaloTower3DProxyBuilder.cc,v 1.10 2009/11/10 20:43:55 amraktad Exp $
//

#include <math.h>
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"
#include "TEveManager.h"
#include "TEveSelection.h"

#include "Fireworks/Calo/plugins/FWCaloTower3DProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"

#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"

//
// constructors , dectructors
//
FWCaloTower3DProxyBuilderBase::FWCaloTower3DProxyBuilderBase() :
   m_caloData(0),
   m_hist(0)
{
}

FWCaloTower3DProxyBuilderBase::~FWCaloTower3DProxyBuilderBase()
{
   m_caloData->DecDenyDestroy();
}
 
//
// member functions
//
void
FWCaloTower3DProxyBuilderBase::addToScene(TEveElement& iContainer, TEveCaloDataHist** iCaloData)
{
   if(0==*iCaloData)
   {
      *iCaloData = new TEveCaloDataHist();

      //Setup for selection
      FWFromTEveCaloDataSelector* sel = new FWFromTEveCaloDataSelector(*iCaloData);
      //make sure it is accessible via the base class
      (*iCaloData)->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
      
      
      //NOTE: must attach a histogram to TEveCaloDataHist before passing TEveCaloDataHist to TEveCalo3D
      // else we get a segmentation fault.
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      TH2F* dummy = new TH2F("background",
                             "background",
                             82, fw3dlego::xbins,
                             72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      Int_t sliceIndex = (*iCaloData)->AddHistogram(dummy);
      (*iCaloData)->RefSliceInfo(sliceIndex).Setup("background", 0., 0);

      TEveCalo3D* calo3d = new TEveCalo3D(*iCaloData);
      calo3d->SetMaxTowerH( 150 );
      calo3d->SetScaleAbs( false );
      calo3d->SetBarrelRadius(129);
      calo3d->SetEndCapPos(310);
      calo3d->SetFrameTransparency(80);
      iContainer.AddElement(calo3d);
   }
   m_caloData = *iCaloData;
   (*iCaloData)->IncDenyDestroy();

}

void
FWCaloTower3DProxyBuilderBase::build(const FWEventItem* iItem,
                                     TEveElementList** product)
{
   m_towers=0;
   iItem->get(m_towers);
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

      FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(m_caloData->GetUserData());
      assert(0!=base);
      FWFromTEveCaloDataSelector* sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
      assert(0!=sel);
      sel->addSliceSelector(m_sliceIndex,FWFromSliceSelector(m_hist,iItem));      
   }
   m_hist->Reset();

   if(0==*product) {
      //NOTE: the base class requires that something gets attached or else model changes willl not
      // be propagated to the base class
      *product = new TEveElementList();
   }
   if(iItem->defaultDisplayProperties().isVisible()) {
      m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower) {
         (m_hist)->Fill(tower->eta(),tower->phi(), getEt(*tower));
      }
   }
   m_caloData->DataChanged();
}

void
FWCaloTower3DProxyBuilderBase::modelChanges(const FWModelIds&, TEveElement* iElements)
{
   applyChangesToAllModels(iElements);
}

void
FWCaloTower3DProxyBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   if(m_caloData && m_towers && item()) {
      m_hist->Reset();

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
         if(0==m_caloData->GetSelectedLevel()) {
            gEve->GetSelection()->AddElement(m_caloData);
         }
      } else {
         if(1==m_caloData->GetSelectedLevel()||2==m_caloData->GetSelectedLevel()) {
            gEve->GetSelection()->RemoveElement(m_caloData);
         }
      }
      
      m_caloData->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      m_caloData->CellSelectionChanged();
	  m_caloData->DataChanged();

   }
}

void
FWCaloTower3DProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
   FW3DDataProxyBuilder::itemBeingDestroyed(iItem);
   if(0!=m_hist) {
      m_hist->Reset();
   }
   if(0 != m_caloData) {
      m_caloData->DataChanged();
   }
}

REGISTER_FW3DDATAPROXYBUILDER(FWECalCaloTower3DProxyBuilder,CaloTowerCollection,"ECal");
REGISTER_FW3DDATAPROXYBUILDER(FWHCalCaloTower3DProxyBuilder,CaloTowerCollection,"HCal");


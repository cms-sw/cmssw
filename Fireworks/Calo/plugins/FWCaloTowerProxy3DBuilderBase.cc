// -*- C++ -*-
// $Id: FWCaloTowerProxy3DBuilderBase.cc,v 1.1 2009/01/14 12:06:45 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "RVersion.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TH2F.h"

// user include files
#include "Fireworks/Calo/plugins/FWCaloTowerProxy3DBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

TEveCaloDataHist* FWCaloTowerProxy3DBuilderBase::m_data = 0;

void FWCaloTowerProxy3DBuilderBase::build(const FWEventItem* iItem, TEveElementList** product)
{
   m_towers=0;
   iItem->get(m_towers);
   if(0==m_towers) {
      if(0 != m_hist) {
         m_hist->Reset();
         if(m_data) {
            m_data->DataChanged();
         }
      }
      return;
   }
   bool newHist = false;
   if ( m_hist == 0 ) {
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      m_hist = new TH2F(m_histName,"CaloTower ECAL Et distribution", 82, fw3dlego::xbins, 72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      newHist = true;
   }
   m_hist->Reset();
   if(iItem->defaultDisplayProperties().isVisible()) {
      for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower) {
         if(m_handleEcal) {
            (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
         } else {
            (m_hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
         }
      }
   }
   if ( ! m_data )  {
      m_data = new TEveCaloDataHist();
      //make sure it does not go away
      m_data->IncRefCount();
   }
   if ( newHist ) {
      m_sliceIndex = m_data->AddHistogram(m_hist);
      m_data->RefSliceInfo(m_sliceIndex).Setup(m_histName, 0., iItem->defaultDisplayProperties().color());
   }

   if ( m_calo3d == 0 ) {
      m_calo3d = new TEveCalo3D(m_data);
      m_calo3d->SetBarrelRadius(129);
      m_calo3d->SetEndCapPos(310);
      // gEve->AddElement(m_calo3d);
      //	(*product)->AddElement(m_calo3d);
   }
   if( *product == 0) {
      *product = new TEveElementList();
      //Since m_calo3d can be shared by multiple proxy builders we have to
      // be sure it gets attached to multiple outputs
      gEve->AddElement(*product);
      gEve->AddElement(m_calo3d, *product);
   }
}

void
FWCaloTowerProxy3DBuilderBase::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   FWRPZDataProxyBuilder::itemBeingDestroyedImp(iItem);
   if(0!= m_hist) { m_hist->Reset(); }
   if(0!= m_data) {m_data->DataChanged();}
}

void
FWCaloTowerProxy3DBuilderBase::modelChanges(const FWModelIds& iIds,
                                          TEveElement* iElements )
{
   applyChangesToAllModels(iElements);
}

void
FWCaloTowerProxy3DBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   if(m_data && m_towers && item()) {
      m_hist->Reset();
      if(item()->defaultDisplayProperties().isVisible()) {

         assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            if(item()->modelInfo(index).displayProperties().isVisible()) {
               if(m_handleEcal) {
                  (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
               } else {
                  (m_hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
               }
            }
         }
      }
      m_data->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      m_data->DataChanged();
   }
}

// -*- C++ -*-
// $Id: ECalCaloTowerProxy3DBuilder.cc,v 1.8 2008/07/17 18:29:28 chrjones Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TH2F.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include "Fireworks/Calo/interface/ECalCaloTowerProxy3DBuilder.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

TEveCaloDataHist*ECalCaloTowerProxy3DBuilder::m_data=0;


void ECalCaloTowerProxy3DBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
/*
   TH2F* hist = 0;
   TEveCaloDataHist* data = 0;
   std::string name = "ecal3D";
   if ( m_calo3d ) data = dynamic_cast<TEveCaloDataHist*>( m_calo3d->GetData() );
   if ( data ) {
      for ( Int_t i = 0; i < data->GetNSlices(); ++i ){
	 TH2F* h = data->RefSliceInfo(i).fHist;
	 if ( ! h ) continue;
	 if ( name == h->GetName() ) {
	    hist = h;
	    break;
	 }
      }
   }
 */
   m_towers=0;
   iItem->get(m_towers);
   if(0==m_towers) return;

   bool newHist = false;
   if ( m_hist == 0 ) {
     Bool_t status = TH1::AddDirectoryStatus();
     TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
     m_hist = new TH2F(histName().c_str(),"CaloTower ECAL Et distribution", 82, fw3dlego::xbins, 72, -M_PI, M_PI);
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
   if ( ! m_data ) m_data = new TEveCaloDataHist();
   if ( newHist ) {
      m_sliceIndex = m_data->AddHistogram(m_hist);
      m_data->RefSliceInfo(m_sliceIndex).Setup(histName().c_str(), 0., iItem->defaultDisplayProperties().color());
   }

   if ( m_calo3d == 0 ) {
      m_calo3d = new TEveCalo3D(m_data);
      m_calo3d->SetBarrelRadius(129);
      m_calo3d->SetEndCapPos(310);
      m_calo3d->IncDenyDestroy(); //Can't allow this to be destroyed
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

std::string
ECalCaloTowerProxy3DBuilder::histName() const
{
   return "ecal3D";
}

void
ECalCaloTowerProxy3DBuilder::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   FWRPZDataProxyBuilder::itemBeingDestroyedImp(iItem);
   m_hist->Reset();
   m_data->DataChanged();
}


void
ECalCaloTowerProxy3DBuilder::modelChanges(const FWModelIds& iIds,
                                          TEveElement* iElements )
{
   applyChangesToAllModels(iElements);
}

void
ECalCaloTowerProxy3DBuilder::applyChangesToAllModels(TEveElement* iElements)
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

REGISTER_FWRPZDATAPROXYBUILDER(ECalCaloTowerProxy3DBuilder,CaloTowerCollection,"ECal");

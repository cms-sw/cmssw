// -*- C++ -*-
// $Id: FWCaloTowerRPZProxyBuilder.cc,v 1.6 2009/10/22 16:22:42 chrjones Exp $
//

// system include files
#include "TEveManager.h"
#include "RVersion.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TH2F.h"
#include "TEveSelection.h"

// user include files
#include "Fireworks/Calo/plugins/FWCaloTowerRPZProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

TEveCaloDataHist* FWCaloTowerRPZProxyBuilderBase::m_data = 0;

FWCaloTowerRPZProxyBuilderBase::FWCaloTowerRPZProxyBuilderBase(bool handleEcal, const char* name):
   FWRPZDataProxyBuilder(),
   m_ownData(kFALSE),
   m_handleEcal(handleEcal),
   m_histName(name),
   m_hist(0),
   m_sliceIndex(-1)
{
   setHighPriority( true );
}

FWCaloTowerRPZProxyBuilderBase::~FWCaloTowerRPZProxyBuilderBase()
{
   // Destructor.

   if( 0 !=m_data && 0 != m_hist) {m_data->DecDenyDestroy();}
}

//______________________________________________________________________________
void FWCaloTowerRPZProxyBuilderBase::build(const FWEventItem* iItem, TEveElementList** product)
{
   m_modelIndexToCellId.clear();
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

   if ( !m_data )  {
      m_data = new TEveCaloDataHist();
   }
   if ( newHist ) {
      m_sliceIndex = m_data->AddHistogram(m_hist);
      m_data->RefSliceInfo(m_sliceIndex).Setup(m_histName, 0., iItem->defaultDisplayProperties().color());
      //make sure it does not go away
      m_data->IncDenyDestroy();
   }
   
   m_hist->Reset();
   m_modelIndexToCellId.reserve(m_towers->size());
   for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower) {
      if(iItem->defaultDisplayProperties().isVisible()) {         
         if(m_handleEcal) {
            (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
         } else {
            (m_hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
         }
      }
      //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
      // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice
      m_modelIndexToCellId.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(),tower->phi()),m_sliceIndex));
      //std::cout <<" index "<<m_modelIndexToCellId.size()-1<<" bin # "<<m_hist->FindBin(tower->eta(),tower->phi())
      //<<" eta "<< tower->eta()<<" phi "<<tower->phi()<<std::endl;
   }
   if ( m_calo3d == 0 ) {
      m_calo3d = new TEveCalo3D(m_data, "RPZCalo3D");
      m_calo3d->SetBarrelRadius(129);
      m_calo3d->SetEndCapPos(310);
      if ( *product == 0)
      {
         *product = new TEveElementList("RPZCalo3DHolder");
         (*product)->AddElement(m_calo3d);
         gEve->AddElement(*product);
      }
   }
}

void
FWCaloTowerRPZProxyBuilderBase::itemBeingDestroyedImp(const FWEventItem* iItem)
{
   FWRPZDataProxyBuilder::itemBeingDestroyedImp(iItem);
   if(0!= m_hist) { m_hist->Reset(); }
   if(0!= m_data) {m_data->DataChanged();}
}

void
FWCaloTowerRPZProxyBuilderBase::modelChanges(const FWModelIds& iIds,
                                             TEveElement* iElements )
{
   applyChangesToAllModels(iElements);
}

void
FWCaloTowerRPZProxyBuilderBase::applyChangesToAllModels(TEveElement* iElements)
{
   if(m_data && m_towers && item()) {
      m_hist->Reset();
      bool somethingSelected = false;
      if(item()->defaultDisplayProperties().isVisible()) {

         assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         //NOTE: technically should only remove the cells for this slice, not all cells
         m_data->GetCellsSelected().clear();
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               if(m_handleEcal) {
                  (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
               } else {
                  (m_hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
               }
               if(info.isSelected()) {
                  m_data->GetCellsSelected().push_back(m_modelIndexToCellId[index]);
                  //std::cout <<"selected "<<index<<" cellID "<<m_modelIndexToCellId[index].fTower<<std::endl;
                  somethingSelected=true;
               }
            }
         }
      }
      if(somethingSelected) {
         if(0==m_data->GetSelectedLevel()) {
            gEve->GetSelection()->AddElement(m_data);
         }
      } else {
         if(0!=m_data->GetSelectedLevel()) {
            gEve->GetSelection()->RemoveElement(m_data);
         }
      }
      m_data->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      m_data->DataChanged();
   }
}


REGISTER_FWRPZDATAPROXYBUILDERBASE(FWECalCaloTowerRPZProxyBuilder,CaloTowerCollection,"ECal");
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWHCalCaloTowerRPZProxyBuilder,CaloTowerCollection,"HCal");

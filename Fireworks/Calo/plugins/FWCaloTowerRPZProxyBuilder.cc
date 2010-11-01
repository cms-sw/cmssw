// -*- C++ -*-
// $Id: FWCaloTowerRPZProxyBuilder.cc,v 1.18 2009/11/14 16:22:37 chrjones Exp $
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
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"

FWCaloTowerRPZProxyBuilderBase::FWCaloTowerRPZProxyBuilderBase(bool handleEcal, const char* name):
   FWRPZDataProxyBuilder(),
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

   if( 0 !=m_data) {m_data->DecDenyDestroy();}
}

//______________________________________________________________________________
void FWCaloTowerRPZProxyBuilderBase::build(const FWEventItem* iItem, TEveElementList** product)
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

      m_sliceIndex = m_data->AddHistogram(m_hist);
      m_data->RefSliceInfo(m_sliceIndex).Setup(m_histName, 0., iItem->defaultDisplayProperties().color());
      FWFromEveSelectorBase* base = reinterpret_cast<FWFromEveSelectorBase*>(m_data->GetUserData());
      assert(0!=base);
      FWFromTEveCaloDataSelector* sel = dynamic_cast<FWFromTEveCaloDataSelector*> (base);
      assert(0!=sel);
      sel->addSliceSelector(m_sliceIndex,FWFromSliceSelector(m_hist,iItem));
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
   if ( *product == 0)
   {
      *product = new TEveElementList("RPZCalo3DHolder");
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
      
      //find all selected cell ids which are not from this FWEventItem and preserve only them
      // do this by moving them to the end of the list and then clearing only the end of the list
      // this avoids needing any additional memory
      TEveCaloData::vCellId_t& selected = m_data->GetCellsSelected();
      //std::cout <<"FWCaloTowerRPZProxyBuilderBase::applyChangesToAllModels "<< selected.size()<<std::endl;
      
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
               if(m_handleEcal) {
				   //std::cout <<"show ecal "<<index<<std::endl;
                  (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
               } else {
				   //std::cout <<"show hcal "<<index<<std::endl;
                  (m_hist)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
               }
               if(info.isSelected()) {
                  //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
                  // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice

                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(),tower->phi()),m_sliceIndex));
               }
            }
         }
      }
      if(!selected.empty()) {
         if(0==m_data->GetSelectedLevel()) {
            gEve->GetSelection()->AddElement(m_data);
         }
      } else {
         if(1==m_data->GetSelectedLevel()||2==m_data->GetSelectedLevel()) {
            gEve->GetSelection()->RemoveElement(m_data);
         }
      }
      m_data->SetSliceColor(m_sliceIndex,item()->defaultDisplayProperties().color());
      m_data->CellSelectionChanged();
	  m_data->DataChanged(); //needed to force it to redraw cells
   }
}

void 
FWCaloTowerRPZProxyBuilderBase::useCalo(TEveCaloDataHist* ioHist) 
{
   m_data = ioHist;
   if(0==m_data->GetUserData()) {
      FWFromTEveCaloDataSelector* sel = new FWFromTEveCaloDataSelector(m_data);
      //make sure it is accessible via the base class
      m_data->SetUserData(static_cast<FWFromEveSelectorBase*>(sel));
   }
   m_data->IncDenyDestroy();
}



REGISTER_FWRPZDATAPROXYBUILDERBASE(FWECalCaloTowerRPZProxyBuilder,CaloTowerCollection,"ECal");
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWHCalCaloTowerRPZProxyBuilder,CaloTowerCollection,"HCal");

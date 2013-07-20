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
// $Id: FWCaloTowerProxyBuilder.cc,v 1.25 2011/02/23 11:34:52 amraktad Exp $
//

// system includes
#include <math.h>

// user includes
#include "TEveCaloData.h"
#include "TEveCalo.h"
#include "TH2F.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "Fireworks/Calo/plugins/FWCaloTowerProxyBuilder.h"
#include "Fireworks/Calo/plugins/FWCaloTowerSliceSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"



//
// constructors , dectructors
//
FWCaloTowerProxyBuilderBase::FWCaloTowerProxyBuilderBase():
m_towers(0),
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
FWCaloTowerProxyBuilderBase::setCaloData(const fireworks::Context&)
{
  m_caloData = context().getCaloData();
}


void
FWCaloTowerProxyBuilderBase::build(const FWEventItem* iItem,
                                   TEveElementList* el, const FWViewContext* ctx)
{
   m_towers=0;
   if (iItem)
   {
      iItem->get(m_towers);
      FWCaloDataProxyBuilderBase::build(iItem, el, ctx);
   }
}

void
FWCaloTowerProxyBuilderBase::itemBeingDestroyed(const FWEventItem* iItem)
{
  
   if(0!=m_hist) {
      m_hist->Reset();
   }
   FWCaloDataProxyBuilderBase::itemBeingDestroyed(iItem);
}

double
wrapPi(double val)
{
   using namespace TMath;

   if (val< -Pi())
   {
      return val += TwoPi();
   }
   if (val> Pi())
   {
      return val -= TwoPi();
   }
   return val;
}

void
FWCaloTowerProxyBuilderBase::fillCaloData()
{
   static float d = 2.5*TMath::Pi()/180;
   m_hist->Reset();

   if (m_towers)
   {
      TEveCaloData::vCellId_t& selected = m_caloData->GetCellsSelected();

      if(item()->defaultDisplayProperties().isVisible()) {
         // assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            const FWEventItem::ModelInfo& info = item()->modelInfo(index);
            if(info.displayProperties().isVisible()) {
               if (tower->ietaAbs() > 39)
               {
                  m_hist->Fill(tower->eta(),wrapPi(tower->phi() - 3*d), getEt(*tower) *0.25);
                  m_hist->Fill(tower->eta(),wrapPi(tower->phi() -   d), getEt(*tower) *0.25);
                  m_hist->Fill(tower->eta(),wrapPi(tower->phi() +   d), getEt(*tower) *0.25);
                  m_hist->Fill(tower->eta(),wrapPi(tower->phi() + 3*d), getEt(*tower) *0.25);
               }
               else if (tower->ietaAbs() > 20)
               {
                  m_hist->Fill(tower->eta(),wrapPi(tower->phi() - d), getEt(*tower) *0.5);
                  m_hist->Fill(tower->eta(),wrapPi(tower->phi() + d), getEt(*tower) *0.5);
               }
               else
               {
                  m_hist->Fill(tower->eta(),tower->phi(), getEt(*tower));
               }
            }
            if(info.isSelected()) {
               //NOTE: I tried calling TEveCalo::GetCellList but it always returned 0, probably because of threshold issues
               // but looking at the TEveCaloHist::GetCellList code the CellId_t is just the histograms bin # and the slice
               // printf("applyChangesToAllModels ...check selected \n");

               if (tower->ietaAbs() > 39)
               {
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(), wrapPi(tower->phi() -3*d)),m_sliceIndex));
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(), wrapPi(tower->phi() -d))  ,m_sliceIndex));
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(), wrapPi(tower->phi() +d))  ,m_sliceIndex));
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(), wrapPi(tower->phi() +3*d)),m_sliceIndex));
               }
               else if (tower->ietaAbs() > 20)
               {
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(), wrapPi(tower->phi() -d)), m_sliceIndex));
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(), wrapPi(tower->phi() +d)), m_sliceIndex));
               }
               else
               {
                  selected.push_back(TEveCaloData::CellId_t(m_hist->FindBin(tower->eta(),tower->phi()),m_sliceIndex));
               }
            }
         }
      }
   }

}

//______________________________________________________________________________
bool
FWCaloTowerProxyBuilderBase::assertCaloDataSlice()
{
   if (m_hist == 0)
   {
      // add new slice
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      m_hist = new TH2F("caloHist",
                        "caloHist",
                        fw3dlego::xbins_n - 1, fw3dlego::xbins,
                        72, -M_PI, M_PI);
      TH1::AddDirectory(status);
      TEveCaloDataHist* ch = static_cast<TEveCaloDataHist*>(m_caloData);
      m_sliceIndex = ch->AddHistogram(m_hist);



      m_caloData->RefSliceInfo(m_sliceIndex).Setup(item()->name().c_str(), 0., 
                                                   item()->defaultDisplayProperties().color(),
                                                   item()->defaultDisplayProperties().transparency());

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
     
      sel->addSliceSelector(m_sliceIndex, new FWCaloTowerSliceSelector(m_hist,item()));
     
      return true;
   }
   return false;
}

REGISTER_FWPROXYBUILDER(FWECalCaloTowerProxyBuilder,CaloTowerCollection,"ECal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits);
REGISTER_FWPROXYBUILDER(FWHCalCaloTowerProxyBuilder,CaloTowerCollection,"HCal",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits );
REGISTER_FWPROXYBUILDER(FWHOCaloTowerProxyBuilder,CaloTowerCollection,"HCal Outer",FWViewType::k3DBit|FWViewType::kAllRPZBits|FWViewType::kAllLegoBits );




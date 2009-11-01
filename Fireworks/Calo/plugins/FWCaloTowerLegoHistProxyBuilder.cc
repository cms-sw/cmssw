// -*- C++ -*-
// $Id: FWCaloTowerLegoHistProxyBuilder.cc,v 1.1 2009/01/19 17:59:12 amraktad Exp $


// system include files
#include "TH2F.h"

#include "Fireworks/Calo/plugins/FWCaloTowerLegoHistProxyBuilder.h"
#include "Fireworks/Core/interface/FW3DLegoEveHistProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

void
FWCaloTowerLegoHistBuilderBase::build(const FWEventItem* iItem,
                                      TH2F** product)
{
   if (0==*product) {
      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      *product = new TH2F(histName(),Form("CaloTower %s Et distribution", histName()),
                          82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
      m_hist = *product;
      TH1::AddDirectory(status);
   }
   (*product)->Reset();
   (*product)->SetFillColor(iItem->defaultDisplayProperties().color());

   m_towers=0;
   iItem->get(m_towers);
   if(0==m_towers) return;
   fillHist();
}


void
FWCaloTowerLegoHistBuilderBase::applyChangesToAllModels()
{
   if(m_towers && item()) {
      m_hist->Reset();
      if(item()->defaultDisplayProperties().isVisible()) {
         assert(item()->size() >= m_towers->size());
         fillHist();
      }
   }
}


//
// Ecal
//

void
FWECalCaloTowerLegoHistBuilder::fillHist()
{
   unsigned int index=0;
   for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
      if(item()->modelInfo(index).displayProperties().isVisible()) {
         (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
      }
   }
}

REGISTER_FW3DLEGODATAPROXYBUILDER(FWECalCaloTowerLegoHistBuilder,CaloTowerCollection,"ECal");


//
// Hcal
//

void
FWHCalCaloTowerLegoHistBuilder::fillHist()
{
   unsigned int index=0;
   for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
      if(item()->modelInfo(index).displayProperties().isVisible()) {
         m_hist->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
      }
   }
}

REGISTER_FW3DLEGODATAPROXYBUILDER(FWHCalCaloTowerLegoHistBuilder,CaloTowerCollection,"HCal");

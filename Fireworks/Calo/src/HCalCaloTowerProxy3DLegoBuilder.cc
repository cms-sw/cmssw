// -*- C++ -*-
//
// Package:     Calo
// Class  :     HCalCaloTowerProxy3DLegoBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: HCalCaloTowerProxy3DLegoBuilder.cc,v 1.8 2008/07/16 13:51:00 dmytro Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Calo/interface/HCalCaloTowerProxy3DLegoBuilder.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HCalCaloTowerProxy3DLegoBuilder::HCalCaloTowerProxy3DLegoBuilder(): m_towers(0),m_hist(0)
{
}

// HCalCaloTowerProxy3DLegoBuilder::HCalCaloTowerProxy3DLegoBuilder(const HCalCaloTowerProxy3DLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

HCalCaloTowerProxy3DLegoBuilder::~HCalCaloTowerProxy3DLegoBuilder()
{
}

//
// assignment operators
//
// const HCalCaloTowerProxy3DLegoBuilder& HCalCaloTowerProxy3DLegoBuilder::operator=(const HCalCaloTowerProxy3DLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   HCalCaloTowerProxy3DLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
HCalCaloTowerProxy3DLegoBuilder::build(const FWEventItem* iItem,
				       TH2F** product)
{
  if (0==*product) {
    Bool_t status = TH1::AddDirectoryStatus();
    TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
    *product = new TH2F("hcalLego","CaloTower HCAL Et distribution",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
     m_hist=*product;
    TH1::AddDirectory(status);
  }
  (*product)->Reset();
  (*product)->SetFillColor(iItem->defaultDisplayProperties().color());

  m_towers=0;
  iItem->get(m_towers);
  if(0==m_towers) return;

  for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower)
    (*product)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());

}

void
HCalCaloTowerProxy3DLegoBuilder::applyChangesToAllModels()
{
   if(m_towers && item()) {
      m_hist->Reset();
      if(item()->defaultDisplayProperties().isVisible()) {

         assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            if(item()->modelInfo(index).displayProperties().isVisible()) {
               m_hist->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());
            }
         }
      }
   }
}

//
// const member functions
//
REGISTER_FW3DLEGODATAPROXYBUILDER(HCalCaloTowerProxy3DLegoBuilder,CaloTowerCollection,"HCal");

//
// static member functions
//

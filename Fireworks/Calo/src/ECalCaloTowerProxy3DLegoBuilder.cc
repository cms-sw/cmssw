// -*- C++ -*-
//
// Package:     Calo
// Class  :     ECalCaloTowerProxy3DLegoBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ECalCaloTowerProxy3DLegoBuilder.cc,v 1.8 2008/07/16 13:51:00 dmytro Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Calo/interface/ECalCaloTowerProxy3DLegoBuilder.h"
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
ECalCaloTowerProxy3DLegoBuilder::ECalCaloTowerProxy3DLegoBuilder() :
m_towers(0)
{
}

// ECalCaloTowerProxy3DLegoBuilder::ECalCaloTowerProxy3DLegoBuilder(const ECalCaloTowerProxy3DLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

ECalCaloTowerProxy3DLegoBuilder::~ECalCaloTowerProxy3DLegoBuilder()
{
}

//
// assignment operators
//
// const ECalCaloTowerProxy3DLegoBuilder& ECalCaloTowerProxy3DLegoBuilder::operator=(const ECalCaloTowerProxy3DLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   ECalCaloTowerProxy3DLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
ECalCaloTowerProxy3DLegoBuilder::build(const FWEventItem* iItem,
				       TH2F** product)
{
  if (0==*product) {
    Bool_t status = TH1::AddDirectoryStatus();
    TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
    *product = new TH2F("ecalLego","CaloTower ECAL Et distribution",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
    m_hist = *product;
    TH1::AddDirectory(status);
  }
  (*product)->Reset();
  (*product)->SetFillColor(iItem->defaultDisplayProperties().color());

   m_towers=0;
  iItem->get(m_towers);
  if(0==m_towers) return;
  for(CaloTowerCollection::const_iterator tower = m_towers->begin();
      tower != m_towers->end(); ++tower) {
     (*product)->Fill(tower->eta(), tower->phi(), tower->emEt());
  }

}


void
ECalCaloTowerProxy3DLegoBuilder::applyChangesToAllModels()
{
   if(m_towers && item()) {
      m_hist->Reset();
      if(item()->defaultDisplayProperties().isVisible()) {

         assert(item()->size() >= m_towers->size());
         unsigned int index=0;
         for(CaloTowerCollection::const_iterator tower = m_towers->begin(); tower != m_towers->end(); ++tower,++index) {
            if(item()->modelInfo(index).displayProperties().isVisible()) {
               (m_hist)->Fill(tower->eta(), tower->phi(), tower->emEt());
            }
         }
      }
   }
}

//
// const member functions
//
REGISTER_FW3DLEGODATAPROXYBUILDER(ECalCaloTowerProxy3DLegoBuilder,CaloTowerCollection,"ECal");

//
// static member functions
//

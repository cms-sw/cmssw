// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWHCalCaloTowerProxy3DLegoBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWHCalCaloTowerProxy3DLegoBuilder.cc,v 1.10 2008/12/03 20:55:42 chrjones Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Core/interface/FW3DLegoEveHistProxyBuilder.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Core/interface/fw3dlego_xbins.h"

class FWHCalCaloTowerProxy3DLegoBuilder : public FW3DLegoEveHistProxyBuilder
{

   public:
      FWHCalCaloTowerProxy3DLegoBuilder();
      virtual ~FWHCalCaloTowerProxy3DLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void applyChangesToAllModels();
      virtual void build(const FWEventItem* iItem,
                         TH2F** product);

      FWHCalCaloTowerProxy3DLegoBuilder(const FWHCalCaloTowerProxy3DLegoBuilder&); // stop default

      const FWHCalCaloTowerProxy3DLegoBuilder& operator=(const FWHCalCaloTowerProxy3DLegoBuilder&); // stop default

      // ---------- member data --------------------------------
      const CaloTowerCollection* m_towers;
      TH2F* m_hist;

};

//
// constructors and destructor
//
FWHCalCaloTowerProxy3DLegoBuilder::FWHCalCaloTowerProxy3DLegoBuilder(): m_towers(0),m_hist(0)
{
}

FWHCalCaloTowerProxy3DLegoBuilder::~FWHCalCaloTowerProxy3DLegoBuilder()
{
}

//
// member functions
//
void
FWHCalCaloTowerProxy3DLegoBuilder::build(const FWEventItem* iItem,
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
FWHCalCaloTowerProxy3DLegoBuilder::applyChangesToAllModels()
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
REGISTER_FW3DLEGODATAPROXYBUILDER(FWHCalCaloTowerProxy3DLegoBuilder,CaloTowerCollection,"HCal");

//
// static member functions
//

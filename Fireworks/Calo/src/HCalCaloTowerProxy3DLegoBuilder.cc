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
// $Id: HCalCaloTowerProxy3DLegoBuilder.cc,v 1.4 2008/03/06 10:17:16 dmytro Exp $
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
HCalCaloTowerProxy3DLegoBuilder::HCalCaloTowerProxy3DLegoBuilder()
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
				       TH2** product)
{
  if (0==*product) {
    *product = new TH2F("hcalLego","CaloTower HCAL Et distribution",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
  }
  (*product)->Reset();
  (*product)->SetFillColor(iItem->defaultDisplayProperties().color());

  const CaloTowerCollection* towers=0;
  iItem->get(towers);
  if(0==towers) {
    std::cout <<"Failed to get CaloTowers"<<std::endl;
    return;
  }

  for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower)
    (*product)->Fill(tower->eta(), tower->phi(), tower->hadEt()+tower->outerEt());

}

//
// const member functions
//
REGISTER_FW3DLEGODATAPROXYBUILDER(HCalCaloTowerProxy3DLegoBuilder,CaloTowerCollection,"HCal");

//
// static member functions
//

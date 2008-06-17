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
// $Id: ECalCaloTowerProxy3DLegoBuilder.cc,v 1.5 2008/06/09 19:54:03 chrjones Exp $
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
ECalCaloTowerProxy3DLegoBuilder::ECalCaloTowerProxy3DLegoBuilder()
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
				       TH2** product)
{
  if (0==*product) {
    Bool_t status = TH1::AddDirectoryStatus();
    TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
    *product = new TH2F("ecalLego","CaloTower ECAL Et distribution",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
    TH1::AddDirectory(status);
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
    (*product)->Fill(tower->eta(), tower->phi(), tower->emEt());

}

//
// const member functions
//
REGISTER_FW3DLEGODATAPROXYBUILDER(ECalCaloTowerProxy3DLegoBuilder,CaloTowerCollection,"ECal");

//
// static member functions
//

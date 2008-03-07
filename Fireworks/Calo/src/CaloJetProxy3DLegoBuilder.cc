// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxy3DLegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetProxy3DLegoBuilder.cc,v 1.4 2008/03/06 10:17:16 dmytro Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Calo/interface/CaloJetProxy3DLegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloJetProxy3DLegoBuilder::CaloJetProxy3DLegoBuilder()
{
}

// CaloJetProxy3DLegoBuilder::CaloJetProxy3DLegoBuilder(const CaloJetProxy3DLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetProxy3DLegoBuilder::~CaloJetProxy3DLegoBuilder()
{
}

//
// assignment operators
//
// const CaloJetProxy3DLegoBuilder& CaloJetProxy3DLegoBuilder::operator=(const CaloJetProxy3DLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   CaloJetProxy3DLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
CaloJetProxy3DLegoBuilder::build(const FWEventItem* iItem, 
				       TH2** product)
{
  if (0==*product) {
    *product = new TH2F("jetsLego","Jets distribution",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
  }
  (*product)->Reset();
  (*product)->SetFillColor(iItem->defaultDisplayProperties().color());
  
  build(iItem,*product,kFALSE);
}

void 
CaloJetProxy3DLegoBuilder::build(const FWEventItem* iItem, 
				 TH2* product,
				 bool selectedFlag )
{
   const reco::CaloJetCollection* jets=0;
   iItem->get(jets);
   if(0==jets) {
      std::cout <<"Failed to get CaloJets"<<std::endl;
      return;
   }
   
   for ( unsigned int i = 0; i < jets->size(); ++i ) {
      // printf("jet pt: %0.2f, eta: %0.2f, phi: %0.2f\n",jets->at(i).pt(), jets->at(i).eta(), jets->at(i).phi());
      if ( ! iItem->modelInfo(i).displayProperties().isVisible() ) continue;
      if ( iItem->modelInfo(i).isSelected() != selectedFlag ) continue;
      std::vector<CaloTowerRef> towers = jets->at(i).getConstituents();
      for ( std::vector<CaloTowerRef>::const_iterator tower = towers.begin();
	    tower != towers.end(); ++tower )
	{
	   // printf("\ttower eta: %0.2f, phi: %0.2f, et: %0.2f, ieta: %d, iphi: %d\n",
	   // (*tower)->eta(), (*tower)->phi(), (*tower)->et(),
	   // product->GetXaxis()->FindFixBin((*tower)->eta()),
	   // product->GetYaxis()->FindFixBin((*tower)->phi()) );
	   if ( dynamic_cast<TH2C*>( product ) )
	     product->Fill((*tower)->eta(), (*tower)->phi());
	   else
	     product->Fill((*tower)->eta(), (*tower)->phi(), 0.01);
	}
   }
}

//
// const member functions
//

//
// static member functions
//

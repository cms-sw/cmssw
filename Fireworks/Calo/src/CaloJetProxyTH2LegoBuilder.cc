// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxyTH2LegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetProxyTH2LegoBuilder.cc,v 1.6 2008/03/07 09:06:48 dmytro Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Calo/interface/CaloJetProxyTH2LegoBuilder.h"
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
CaloJetProxyTH2LegoBuilder::CaloJetProxyTH2LegoBuilder()
  :m_product(0)
{
}

// CaloJetProxyTH2LegoBuilder::CaloJetProxyTH2LegoBuilder(const CaloJetProxyTH2LegoBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetProxyTH2LegoBuilder::~CaloJetProxyTH2LegoBuilder()
{
}

//
// assignment operators
//
// const CaloJetProxyTH2LegoBuilder& CaloJetProxyTH2LegoBuilder::operator=(const CaloJetProxyTH2LegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   CaloJetProxyTH2LegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
CaloJetProxyTH2LegoBuilder::build(const FWEventItem* iItem, 
				       TH2** product)
{
  if (0==*product) {
     TH2F* h = new TH2F("jetsLego","Jets distribution",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
     m_product = h;
     *product = h;
  }
  (*product)->Reset();
  (*product)->SetFillColor(iItem->defaultDisplayProperties().color());
  
  build(iItem,*product,kFALSE);
}

void 
CaloJetProxyTH2LegoBuilder::build(const FWEventItem* iItem, 
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

void CaloJetProxyTH2LegoBuilder::message( int type, int xbin, int ybin )
{
   const FWEventItem* iItem = getItem();
   const reco::CaloJetCollection* jets=0;
   iItem->get(jets);
   if(0==jets) {
      std::cout <<"Failed to get CaloJets"<<std::endl;
      return;
   }
   if ( ! m_product ) return;
   
   // check if any jets contibute to the selected bin
   // and if not, change message type to zero - unselect.
   if ( type && m_product->GetBinContent(xbin,ybin) < 1e-9 ) type = 0; 
   
   for ( unsigned int i = 0; i < jets->size(); ++i ) {
      if ( ! iItem->modelInfo(i).displayProperties().isVisible() ) continue;
      if ( type == 0 ) {
	 iItem->unselect(i);
	 continue;
      }
      std::vector<CaloTowerRef> towers = jets->at(i).getConstituents();
      bool selected = false;
      for ( std::vector<CaloTowerRef>::const_iterator tower = towers.begin();
	    tower != towers.end(); ++tower )
	{
	   if ( m_product->GetXaxis()->FindFixBin((*tower)->eta()) == xbin &&
		m_product->GetYaxis()->FindFixBin((*tower)->phi()) == ybin )
	     {
		selected = true;
		break;
	     }
	}
      if ( selected && iItem->modelInfo(i).isSelected() ) continue;
      if ( ! selected && ! iItem->modelInfo(i).isSelected() ) continue;
      if ( ! selected && iItem->modelInfo(i).isSelected() ) iItem->unselect(i);
      if ( selected && ! iItem->modelInfo(i).isSelected() ) iItem->select(i);
   }
   
}


//
// const member functions
//

//
// static member functions
//

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
// $Id: CaloJetProxy3DLegoBuilder.cc,v 1.1 2008/01/07 05:48:45 chrjones Exp $
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
				       TH2F** product)
{
  if (0==*product) {
    *product = new TH2F("jetsLego","Jets distribution",
			78, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
  }
  (*product)->Reset();
  (*product)->SetFillColor(iItem->displayProperties().color());

  const reco::CaloJetCollection* jets=0;
  iItem->get(jets);
  if(0==jets) {
    std::cout <<"Failed to get CaloJets"<<std::endl;
    return;
  }

  double minJetEt = 15; // GeV
  double coneSize = 0.5; // jet cone size
  for ( int ix = 1; ix <= (*product)->GetNbinsX(); ++ix ) {
    for ( int iy = 1; iy <= (*product)->GetNbinsY(); ++iy ) {
      for(reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
	if ( jet->et() > minJetEt &&
	     deltaR( jet->eta(), jet->phi(), 
		     (*product)->GetXaxis()->GetBinCenter(ix),
		     (*product)->GetYaxis()->GetBinCenter(iy) ) < 
	     coneSize + sqrt( pow((*product)->GetXaxis()->GetBinWidth(ix),2) +
			      pow((*product)->GetYaxis()->GetBinWidth(iy),2) ) ) {
	  (*product)->SetBinContent(ix, iy, 0.1);
	}
      }
    }
  }
}

double 
CaloJetProxy3DLegoBuilder::deltaR( double eta1, double phi1, double eta2, double phi2 )
{
   double dEta = eta2-eta1;
   double dPhi = fabs(phi2-phi1);
   if ( dPhi > 3.1416 ) dPhi = 2*3.1416 - dPhi;
   return sqrt(dPhi*dPhi+dEta*dEta);
}

//
// const member functions
//

//
// static member functions
//

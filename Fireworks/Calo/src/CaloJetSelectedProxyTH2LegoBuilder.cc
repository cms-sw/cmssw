// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetSelectedProxyTH2LegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetSelectedProxyTH2LegoBuilder.cc,v 1.1 2008/03/20 09:39:26 dmytro Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Calo/interface/CaloJetSelectedProxyTH2LegoBuilder.h"
#include "Fireworks/Calo/interface/CaloJetProxyTH2LegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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
CaloJetSelectedProxyTH2LegoBuilder::CaloJetSelectedProxyTH2LegoBuilder()
{
}

// CaloJetSelectedProxyTH2LegoBuilder::CaloJetSelectedProxyTH2LegoBuilder(const CaloJetSelectedProxyTH2LegoBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetSelectedProxyTH2LegoBuilder::~CaloJetSelectedProxyTH2LegoBuilder()
{
}

//
// assignment operators
//
// const CaloJetSelectedProxyTH2LegoBuilder& CaloJetSelectedProxyTH2LegoBuilder::operator=(const CaloJetSelectedProxyTH2LegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   CaloJetSelectedProxyTH2LegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
CaloJetSelectedProxyTH2LegoBuilder::build(const FWEventItem* iItem, 
				       TH2** product)
{
  if (0==*product) {
    *product = new TH2C("jetsLegoSelected","Jets distribution (selected)",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
  }
  (*product)->Reset();
  
  CaloJetProxyTH2LegoBuilder::build(iItem,*product,kTRUE);
}

//
// const member functions
//

//
// static member functions
//

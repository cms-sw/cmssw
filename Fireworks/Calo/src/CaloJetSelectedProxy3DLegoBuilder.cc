// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetSelectedProxy3DLegoBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: CaloJetSelectedProxy3DLegoBuilder.cc,v 1.3 2008/02/03 02:43:53 dmytro Exp $
//

// system include files
#include "TH2F.h"


// user include files
#include "Fireworks/Calo/interface/CaloJetSelectedProxy3DLegoBuilder.h"
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
CaloJetSelectedProxy3DLegoBuilder::CaloJetSelectedProxy3DLegoBuilder()
{
}

// CaloJetSelectedProxy3DLegoBuilder::CaloJetSelectedProxy3DLegoBuilder(const CaloJetSelectedProxy3DLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

CaloJetSelectedProxy3DLegoBuilder::~CaloJetSelectedProxy3DLegoBuilder()
{
}

//
// assignment operators
//
// const CaloJetSelectedProxy3DLegoBuilder& CaloJetSelectedProxy3DLegoBuilder::operator=(const CaloJetSelectedProxy3DLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   CaloJetSelectedProxy3DLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
CaloJetSelectedProxy3DLegoBuilder::build(const FWEventItem* iItem, 
				       TH2** product)
{
  if (0==*product) {
    *product = new TH2C("jetsLegoSelected","Jets distribution (selected)",
			82, fw3dlego::xbins, 72/legoRebinFactor(), -3.1416, 3.1416);
  }
  (*product)->Reset();
  
  CaloJetProxy3DLegoBuilder::build(iItem,*product,kTRUE);
}

//
// const member functions
//

//
// static member functions
//

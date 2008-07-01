// -*- C++ -*-
//
// Package:     Calo
// Class  :     HCalCaloTowerProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: HCalCaloTowerProxyRhoPhiZ2DBuilder.cc,v 1.7 2008/06/23 22:56:53 dmytro Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"

// user include files
#include "Fireworks/Calo/interface/HCalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayEvent.h"

#include "Fireworks/Core/interface/FWRhoPhiZView.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HCalCaloTowerProxyRhoPhiZ2DBuilder::HCalCaloTowerProxyRhoPhiZ2DBuilder()
{
   setHighPriority( true );
}

// HCalCaloTowerProxyRhoPhiZ2DBuilder::HCalCaloTowerProxyRhoPhiZ2DBuilder(const HCalCaloTowerProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

HCalCaloTowerProxyRhoPhiZ2DBuilder::~HCalCaloTowerProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
HCalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   ECalCaloTowerProxyRhoPhiZ2DBuilder::buildCalo(iItem, product, "hcalRhoPhi", m_caloRhoPhi, false);
}

void 
HCalCaloTowerProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   ECalCaloTowerProxyRhoPhiZ2DBuilder::buildCalo(iItem, product, "hcalRhoZ", m_caloRhoZ, false);
}

//
// const member functions
//
REGISTER_FWRPZ2DDATAPROXYBUILDER(HCalCaloTowerProxyRhoPhiZ2DBuilder,CaloTowerCollection,"HCalRhoZSeparate");
//
// static member functions
//

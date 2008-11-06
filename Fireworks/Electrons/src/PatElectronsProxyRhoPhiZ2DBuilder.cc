// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatElectronsProxyRhoPhiZ2DBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: PatElectronsProxyRhoPhiZ2DBuilder.cc,v 1.2 2008/11/04 20:29:25 amraktad Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrack.h"
#include "TEveCompound.h"
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

// user include files
#include "Fireworks/Electrons/interface/PatElectronsProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Electrons/interface/ElectronsProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PatElectronsProxyRhoPhiZ2DBuilder::PatElectronsProxyRhoPhiZ2DBuilder()
{
}

// PatElectronsProxyRhoPhiZ2DBuilder::PatElectronsProxyRhoPhiZ2DBuilder(const PatElectronsProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

PatElectronsProxyRhoPhiZ2DBuilder::~PatElectronsProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void
PatElectronsProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
     TEveElementList* tList = *product;

     // printf("calling PatElectronsProxyRhoPhiZ2DBuilder::buildRhiPhi\n");
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Electron RhoPhi",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     // get electrons
     const std::vector<pat::Electron> *electrons = 0;
     iItem->get(electrons);
     if (electrons == 0) return;
     fw::NamedCounter counter("electron");
     // loop over electrons
     for (std::vector<pat::Electron>::const_iterator electron = electrons->begin();
	  electron != electrons->end(); ++electron,++counter) {
	ElectronsProxyRhoPhiZ2DBuilder::buildElectronRhoPhi(iItem, &*electron, tList, counter);
     }
}

void
PatElectronsProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					  TEveElementList** product)
{
     // printf("calling PatElectronsProxyRhoPhiZ2DBuilder::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Electron RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     const std::vector<pat::Electron> *electrons = 0;
     iItem->get(electrons);
     if (electrons == 0) return;
     fw::NamedCounter counter("electron");
     for (std::vector<pat::Electron>::const_iterator electron = electrons->begin();
	  electron != electrons->end(); ++electron, ++counter) {
	ElectronsProxyRhoPhiZ2DBuilder::buildElectronRhoZ(iItem, &*electron, tList, counter);
     }
}

REGISTER_FWRPZ2DDATAPROXYBUILDER(PatElectronsProxyRhoPhiZ2DBuilder,std::vector<pat::Electron>,"PatElectrons");
